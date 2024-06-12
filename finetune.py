import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra 
import transformers

from data_module import QADataset, custom_data_collator
from trainer import CustomTrainer
from util import SaveTrainingAndEvaluateCallback


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    #Load the base model you want to finetune
    pretrained_model = cfg.pretrained_model_path
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model, 
        use_flash_attention_2=cfg.flash_attention2, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code = True
        )

    #enable gradient checkpointing
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=cfg.ft.LoRA_r, 
        lora_alpha=cfg.ft.LoRA_alpha,
        lora_dropout=cfg.ft.LoRA_dropout,
        target_modules=['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj'], 
        bias="none", 
        task_type="CAUSAL_LM"
        )

    #Wrap the base model and peft_config to create a PeftModel
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    
    
    #load data
    torch_format_dataset = QADataset(data_path = cfg.data_path,
                                     tokenizer=tokenizer,
                                     configs = cfg, 
                                     max_length=max_length, 
                                     split="train",
                                     question_key='question', 
                                     answer_key='answer',
                                     )

    #setup a TrainingArguments class with some training hyperparameters
    batch_size = cfg.ft.batch_size
    gradient_accumulation_steps = cfg.ft.gradient_accumulation_steps
    max_steps = int(cfg.ft.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps)
    print(f"max_steps: {max_steps}")    
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps)

    training_args = transformers.TrainingArguments(
        output_dir=cfg.ft.save_dir,
        learning_rate=cfg.ft.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=cfg.ft.weight_decay,
        evaluation_strategy="no",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, max_steps//10),
        max_steps=max_steps,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        logging_dir=f'{cfg.ft.save_dir}/logs',
        save_steps=steps_per_epoch,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
        compute_metrics=None,
        callbacks=[SaveTrainingAndEvaluateCallback(save_path=f'{cfg.ft.save_dir}/log.txt')],
    )
    
    model.config.use_cache = False
    trainer.train()

    #save model
    model = model.merge_and_unload()
    print(f'model merged and unloaded at {cfg.ft.save_dir}')
    model.save_pretrained(cfg.ft.save_dir)

if __name__ == "__main__":
    main()
