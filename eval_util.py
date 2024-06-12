from tqdm import tqdm
import torch
import evaluate
from rouge_score import rouge_scorer

from data_module import get_batch_loss


def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores['rouge1'].recall)
        rougeL_recall.append(rouge_scores['rougeL'].recall)

    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}


def get_all_evals(cfg, model, tokenizer, eval_dataloader):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    mrr_list = []
    hit_rate_list = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt, scores = run_generation(cfg, batch, model, tokenizer=tokenizer)
            mrr_per_batch, hit_rate_per_batch = compute_MRR(scores, gt, tokenizer)
            mrr_list.extend(mrr_per_batch)
            hit_rate_list.extend(hit_rate_per_batch)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
        
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        probabilities = torch.softmax(outputs.logits, dim=-1)
        
        # Find the maximum probability and its corresponding token index for each position in the sequence
        # log the probaility of outputs
        max_probs, _ = torch.max(probabilities, dim=-1)
        num_token_gt = (batch['labels']!=-100).sum(-1)
        probs = [sum(max_probs[idx,:v]).item() for idx, v in enumerate(num_token_gt)]

        probs = [p/v.item() for p, v in zip(probs, num_token_gt)]

        eval_logs['gt_loss_per_token'] = eval_logs.get('gt_loss_per_token', []) + (gt_loss/num_token_gt).cpu().numpy().tolist()
        eval_logs['gt_loss'] = eval_logs.get('gt_loss', []) + gt_loss.tolist()
        eval_logs['num_token_gt'] = eval_logs.get('num_token_gt', []) + num_token_gt.tolist()
        eval_logs['probs'] = eval_logs.get('probs', []) + probs
        eval_logs['mrr'] = eval_logs.get('mrr', []) + mrr_list
        eval_logs['hit_rate'] = eval_logs.get('hit_rate', []) + hit_rate_list

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))
    eval_logs['generated_text'] = list(zip(input_strings, gen_outputs,ground_truths))
    
    return eval_logs


def run_generation(cfg, batch, model, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]" 
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    if cfg.model_family == 'llama2-7b-chat' or 'mistral-7b-instruct':
        input_strings = [s + split_symbol for s in input_strings]
    
    #tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(
        input_strings, 
        add_special_tokens=True, 
        return_tensors='pt', 
        padding=True
        ).to(model.device)
    
    #generate
    out = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        max_length=cfg.eval.generation.max_length, 
        max_new_tokens=cfg.eval.generation.max_new_tokens, 
        do_sample=False, 
        num_beams=1,
        num_return_sequences=1,
        use_cache=True, 
        pad_token_id=left_pad_tokenizer.eos_token_id,
        output_scores = True, # return logits
        return_dict_in_generate = True
        )
    strs = left_pad_tokenizer.batch_decode(out.sequences[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)    
    scores = out.scores #tuple of tensors (for each generation step) with shape [16, 32000]
    
    return input_strings, strs, ground_truth, scores

def compute_MRR(scores, gt, tokenizer):
    ## gt is a list with length of batch size
    MRR_res = []
    hit_rate = []

    # Convert scores as tuple to torch tensors
    # Initialize an empty tensor of the desired shape, filled with zeros
    score_size = scores[0].shape[0]
    vocab_size = scores[0].shape[1]
    logits = torch.zeros(score_size, 512, vocab_size, device='cuda')

    # Iterate over the tuple of tensors and assign each to the correct position in the combined tensor
    for i, score_tensor in enumerate(scores):
        #print(f"Tensor {i}: shape {score_tensor.shape}, device {score_tensor.device}")
        logits[:, i, :] = score_tensor
    probabilities = torch.nn.functional.softmax(logits, dim=-1) # torch.Size([16, 512, 32000])
    for i in range(len(gt)):
        probs_per_gt = probabilities[i] #torch.Size([512, 32000])
        #reciprocal rank for each ground truth
        reciprocal_ranks = []
        hit_check = []

        # Tokenize the ground truth
        gt_indices = tokenizer.encode(gt[i], add_special_tokens=False)

        for j, gt_index in enumerate(gt_indices):
            # Get the probability distribution for the current token
            probs = probs_per_gt[j] #len = 32000
            sorted_indices = probs.argsort(descending=True)
            # Find the rank of the current token
            positions = (sorted_indices == gt_index).nonzero()
            rank = positions[0].item()+1
            # Calculate reciprocal rank
            reciprocal_rank = 1.0 / rank
            reciprocal_ranks.append(reciprocal_rank)
            # Calculate hit rate
            if rank <= 100:
                hit_check.append(1)
            else:
                hit_check.append(0)
        MRR_res.append(sum(reciprocal_ranks) / len(reciprocal_ranks))
        hit_rate.append(sum(hit_check) / len(hit_check))
    return MRR_res, hit_rate
    