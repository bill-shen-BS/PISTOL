import json
import torch
from torch.utils.data import DataLoader
import os, hydra
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel

from data_module import QADataset, QAForgetEdgeDataset, QARetainedEdgeDataset, QAIndependentSalesDataset, QAIndependentEmploymentDataset, QAFactualDataset, custom_data_collator
from eval_util import get_all_evals


def get_dataloader(cfg, data_path, tokenizer, eval_target):
    if eval_target == "ft":
        torch_format_dataset = QADataset( 
                data_path, 
                tokenizer=tokenizer,
                configs = cfg, 
                max_length=512, 
                split="train", 
            )
    elif eval_target == "forget_edge":
        torch_format_dataset = QAForgetEdgeDataset(
                data_path, 
                tokenizer=tokenizer,
                configs = cfg,
                max_length=512, 
                split = "train", 
        )
    elif eval_target == "all_retained_edge":
        torch_format_dataset = QARetainedEdgeDataset(
                data_path, 
                tokenizer=tokenizer,
                configs = cfg,
                max_length=512, 
                split = "train", 
        )
    elif eval_target == "indepenent_sales_edge":
        torch_format_dataset = QAIndependentSalesDataset(
                data_path, 
                tokenizer=tokenizer,
                configs = cfg,
                max_length=512, 
                split = "train", 
        )
    elif eval_target == "indepedent_employment_edge":
        torch_format_dataset = QAIndependentEmploymentDataset(
                data_path, 
                tokenizer=tokenizer,
                configs = cfg,
                max_length=512, 
                split = "train", 
        )
    elif eval_target == "factual_data":
        torch_format_dataset = QAFactualDataset(
                data_path, 
                tokenizer=tokenizer,
                configs = cfg,
                max_length=512, 
                split = "train", 
        )            
    else:
        raise ValueError(f"Invalid eval_target")
    
    torch_format_dataset.data = torch_format_dataset.data.select(range(min(200, len(torch_format_dataset.data))))
        
    eval_dataloader = DataLoader(
        torch_format_dataset, 
        batch_size=cfg.eval.batch_size, 
        shuffle=False, 
        collate_fn=custom_data_collator
        )
    return eval_dataloader

def custom_evaluate(cfg, data_path, tokenizer, model, eval_target):
    eval_dataloader = get_dataloader(
        cfg=cfg,
        data_path=data_path,
        tokenizer=tokenizer,
        eval_target = eval_target
    )
    model.to("cuda")
    model.eval()
    eval_logs = get_all_evals(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=eval_dataloader,
    )

    # get the average of the input instead of full list
    for k, v in eval_logs.items():
        if not k == 'generated_text':
            if len(v) > 0:
                eval_logs[k] = sum(v)/len(v)
            else:
                eval_logs[k] = 'N/A'
    return eval_logs

# load ft model
def ft_model_call(ft_model_config, ft_model, config):
    model_config = AutoConfig.from_pretrained(
        ft_model_config, 
        use_flash_attention_2=config.flash_attention2,
        trust_remote_code = True, 
        )    
    
    model = AutoModelForCausalLM.from_pretrained(
        ft_model, 
        config=model_config, 
        use_flash_attention_2=config.flash_attention2,
        torch_dtype=torch.bfloat16, 
        trust_remote_code = True, 
        )
    return model

# load forget model
def forget_model_call(ft_model_config, ft_model, adapter, config):
    model_config = AutoConfig.from_pretrained(
        ft_model_config, 
        use_flash_attention_2=config.flash_attention2,
        trust_remote_code = True, 
        )    
    
    base_model = AutoModelForCausalLM.from_pretrained(
        ft_model, 
        config=model_config, 
        use_flash_attention_2=config.flash_attention2,
        torch_dtype=torch.bfloat16, 
        trust_remote_code = True, 
        )
    
    model = PeftModel.from_pretrained(
        base_model, 
        adapter,
        use_flash_attention_2=config.flash_attention2,
        torch_dtype=torch.bfloat16, 
        trust_remote_code = True, 
        )
    return model
    
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    
    pretrained_model = cfg.pretrained_model_path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    #load model for eval
    if cfg.eval.eval_type == "ft":
        ft_model_config = cfg.ft.save_dir
        ft_model = cfg.ft.save_dir
        model = ft_model_call(ft_model_config, ft_model, cfg)
        eval_logs = custom_evaluate(
            cfg=cfg, 
            data_path=cfg.data_path,
            tokenizer=tokenizer, 
            model=model,
            eval_target = 'ft'
            )
        save_dir = cfg.eval.ft_save_dir
        save_filename = "eval.json"
        save_path = os.path.join(save_dir, save_filename) 
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(eval_logs, f, indent=4)

    elif cfg.eval.eval_type == "forget":
        ft_model_config = cfg.ft.save_dir
        ft_model = cfg.ft.save_dir
        adapter = cfg.forget.save_dir
        model = forget_model_call(ft_model_config, ft_model, adapter, cfg)
        
        eval_logs_forget_edge = custom_evaluate(
            cfg=cfg, 
            data_path=cfg.data_path,
            tokenizer=tokenizer, 
            model=model,
            eval_target= 'forget_edge'
            )
        eval_logs_all_retained_edge = custom_evaluate(
            cfg=cfg, 
            data_path=cfg.data_path,
            tokenizer=tokenizer, 
            model=model,
            eval_target= 'all_retained_edge'
            ) 
        
        if len(cfg.independent_sales_edge) > 0 and len(cfg.indepedent_employment_edge) > 0:
            eval_logs_indepedent_sales_edge = custom_evaluate(
                cfg=cfg, 
                data_path=cfg.data_path,
                tokenizer=tokenizer, 
                model=model,
                eval_target= 'indepenent_sales_edge'
                )
            eval_logs_indepedent_employment_edge = custom_evaluate(
                cfg=cfg, 
                data_path=cfg.data_path,
                tokenizer=tokenizer, 
                model=model,
                eval_target= 'indepedent_employment_edge'
                )        
        else:
            eval_logs_indepedent_sales_edge = "N/A"
            eval_logs_indepedent_employment_edge = "N/A"

        eval_logs_factual_data = custom_evaluate(
            cfg=cfg, 
            data_path="data/factual_data.json",
            tokenizer=tokenizer, 
            model=model,
            eval_target='factual_data'
            )   
        
        eval_logs = {
            "forget": eval_logs_forget_edge, 
            "all_retained_edge": eval_logs_all_retained_edge,
            "retained_indepenent_sales_edge": eval_logs_indepedent_sales_edge,
            "retained_indepedent_employment_edge": eval_logs_indepedent_employment_edge,   
            "factual_data": eval_logs_factual_data,
        }       
        save_dir = cfg.eval.forget_save_dir
        save_filename = "eval.json"
        save_path = os.path.join(save_dir, save_filename) 
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(eval_logs, f, indent=4)


if __name__ == "__main__":
    main()
