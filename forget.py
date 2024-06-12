import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra 
import transformers

from data_module import QAForgetDataset, QAForgetDatasetDPO, custom_data_collator_forget, custom_data_collator_forget_dpo
from trainer import CustomTrainerForgetting


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    pretrained_model = cfg.pretrained_model_path

    #Load the finetuned model
    ft_model = cfg.ft.save_dir     
    model = AutoModelForCausalLM.from_pretrained(
        ft_model, 
        use_flash_attention_2=cfg.flash_attention2, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code = True
        )

    #load oracle model if KL used in forgetting
    oracle_model = None
    if cfg.forget.forget_loss == "KL" or "dpo":
            oracle_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model, 
                use_flash_attention_2=cfg.flash_attention2, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code = True
                )

    #enable gradient checkpointing
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=cfg.forget.LoRA_r, 
        lora_alpha=cfg.forget.LoRA_alpha,
        lora_dropout=cfg.forget.LoRA_dropout,
        target_modules=['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj'], 
        bias="none", 
        task_type="CAUSAL_LM"
        )

    ##Wrap the finetuned model and peft_config to create a PeftModel
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    #load tokenizer from the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    
    #load data
    if cfg.forget.forget_loss == "dpo":
        torch_format_dataset = QAForgetDatasetDPO(
                                        data_path=cfg.data_path,
                                        tokenizer=tokenizer,
                                        configs = cfg,
                                        max_length=max_length, 
                                        split="train"
                                        )
    else:
        torch_format_dataset = QAForgetDataset(
                                        data_path=cfg.data_path,
                                        tokenizer=tokenizer,
                                        configs = cfg,
                                        max_length=max_length, 
                                        split="train"
                                        )

    #setup a TrainingArguments class with some training hyperparameters
    batch_size = cfg.forget.batch_size
    gradient_accumulation_steps = cfg.forget.gradient_accumulation_steps
    max_steps = int(cfg.forget.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps)
    print(f"max_steps: {max_steps}")    
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps)

    training_args = transformers.TrainingArguments(
        output_dir=cfg.forget.save_dir,
        learning_rate=cfg.forget.lr, 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=cfg.forget.weight_decay,
        evaluation_strategy="no",
        eval_steps=max(1,max_steps//20),
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, max_steps//10),
        max_steps=max_steps,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        logging_dir=f'{cfg.forget.save_dir}/logs',
        save_steps=steps_per_epoch, 
    )


    if cfg.forget.forget_loss == "dpo":
        trainer = CustomTrainerForgetting(
            model=model,
            args=training_args,
            train_dataset=torch_format_dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=custom_data_collator_forget_dpo,
            oracle_model = oracle_model,
            forget_loss = cfg.forget.forget_loss,
            cfg = cfg,
            compute_metrics=None, 
        )
    else:        
        trainer = CustomTrainerForgetting(
            model=model,
            args=training_args,
            train_dataset=torch_format_dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=custom_data_collator_forget,
            oracle_model = oracle_model,
            forget_loss = cfg.forget.forget_loss, 
            cfg = cfg,
            compute_metrics=None, 
        )
    
    model.config.use_cache = False
    trainer.train()

    #save LoRA adapter
    model.save_pretrained(cfg.forget.save_dir)
    print(f'Forget LoRA adapter saved at {cfg.forget.save_dir}')

if __name__ == "__main__":
    main()