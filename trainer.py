import torch
from transformers import Trainer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F
import json

from data_module import get_batch_loss
from eval_util import get_all_evals
from eval import get_dataloader  


def printll(name, inp):
    print(name, [round(x, 4) for x in inp])


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Create a DataLoader for the evaluation dataset
        eval_dataloader = DataLoader(
            self.eval_dataset, 
            shuffle=False, 
            batch_size=self.args.eval_batch_size, 
            collate_fn=self.data_collator,
            )
        
        self.model.eval()
        all_predictions = []
        all_labels = []

        #load batch
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = move_to_device(batch, self.model.device)
            
            with torch.no_grad():
                loss, logits, labels = self.prediction_step(
                    self.model, 
                    batch, 
                    prediction_loss_only=True, 
                    ignore_keys=ignore_keys
                    )
           # Accumulate predictions and labels
            all_predictions.extend(logits)
            all_labels.extend(labels) 

        # Convert each tensor in the list to CPU and then to a NumPy array
        all_predictions_np = np.array([pred.cpu().numpy() for pred in all_predictions])
        all_lables_np = np.array([labels.cpu().numpy() for labels in all_labels])

        # Call compute_metrics with all data
        metrics = self.compute_metrics(all_predictions_np, all_lables_np)
        print(metrics)
        # Example of manually updating the state and logging the metrics
        self.log(metrics)
        
        # Manually trigger the on_evaluate callback at the end of evaluation
        self.control = self.callback_handler.on_evaluate(self.args, 
                                                         self.state, 
                                                         self.control, 
                                                         metrics)
        return metrics


class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        if self.oracle_model is not None:
            self.oracle_model.to('cuda')
        self.cfg = kwargs.pop('cfg')
        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss
        
        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss
      
        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
          
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])
            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
            retain_loss = torch.nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        elif self.loss_type == "dpo":
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits
            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)
          
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
            print(loss.item())
            outputs = forget_outputs
      
        return (loss, outputs) if return_outputs else loss


    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        forget_inputs, retain_input = inputs
        input_ids, labels, attention_mask = forget_inputs
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Prepare forget dataset
        eval_forget_dataloader = get_dataloader(
            cfg=self.cfg,
            data_path=self.cfg.data_path,
            tokenizer=self.tokenizer,
            eval_target = 'forget_edge'
        )

        # Prepare retain dataset
        eval_retain_dataloader = get_dataloader(
            cfg=self.cfg,
            data_path=self.cfg.data_path,
            tokenizer=self.tokenizer,
            eval_target = 'all_retain_edge'
        )   
        
        self.model.eval()

        eval_logs_forget = get_all_evals(
            cfg=self.cfg,
            model=self.model,
            tokenizer=self.tokenizer,
            eval_dataloader=eval_forget_dataloader,
        )
        for k, v in eval_logs_forget.items():
            if not k == 'generated_text':
                eval_logs_forget[k] = sum(v)/len(v)

        eval_logs_retain = get_all_evals(
            cfg=self.cfg,
            model=self.model,
            tokenizer=self.tokenizer,
            eval_dataloader=eval_retain_dataloader,
        )
        for k, v in eval_logs_retain.items():
            if not k == 'generated_text':
                eval_logs_retain[k] = sum(v)/len(v)

        eval_logs_combined = {
            'eval_logs_forget': eval_logs_forget,
            'eval_logs_retain': eval_logs_retain,
        }

        current_step = self.state.global_step
        save_filename = f"{self.cfg.forget.save_dir}/eval_logs_{current_step}.json"
        with open(save_filename, "w") as f:
            json.dump(eval_logs_combined, f, indent=4)


def compute_metrics(all_predictions,all_labels):
    logits, labels = torch.from_numpy(all_predictions), torch.from_numpy(all_labels)
    preds = torch.from_numpy(all_predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc.item(), "eval loss": loss.item()}


def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))
    return loss

def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    else:
        raise TypeError("Object must be a tensor or a collection of tensors.")
