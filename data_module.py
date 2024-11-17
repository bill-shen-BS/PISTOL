import json
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from datasets import load_dataset
from datasets import Dataset as HFDataset

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

LLAMA2_CHAT_TEMPLATE = "[INST] {instruction} [/INST]"

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer, configs):
    if configs['model_family'] == "llama3-8b-instruct":
        new_question = LLAMA3_CHAT_TEMPLATE.format(instruction=question)
    elif configs['model_family'] == "Qwen2-7B-Instruct":
        new_question = QWEN_CHAT_TEMPLATE.format(instruction=question)
    elif configs['model_family'] == "llama2-7b-chat":
        new_question = LLAMA2_CHAT_TEMPLATE.format(instruction=question)
    elif configs['model_family'] == "gemma-7b-it":
        new_question = GEMMA_CHAT_TEMPLATE.format(instruction=question)
    else:
        # question_start_token =  configs['question_start_tag']
        # question_end_token = configs['question_end_tag']
        # answer_token = configs['answer_tag']
        # new_question = question_start_token + question + question_end_token
        # new_answer = answer_token + answer
        # full_text = new_question + new_answer
        raise ValueError(f"Invalid model_family")
    
    full_text = new_question + answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


def dataset_format_converstion(data_path):
    with open(data_path, 'r') as f:
        all_QA_list = json.load(f)    
    all_QA_dict = {}
    for item in all_QA_list:
        edge = item["edge"]
        # Remove the "edge" key for the new format
        item_dict = {"question": item["question"], "answer": item["answer"]}
        # Check if the edge key exists in the result dictionary
        if edge not in all_QA_dict:
            all_QA_dict[edge] = []
        # Append the item dictionary to the corresponding edge list
        all_QA_dict[edge].append(item_dict)
    return all_QA_dict

class QADataset(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 configs,
                 max_length=512, 
                 split = None, 
                 question_key='question', 
                 answer_key='answer'
                 ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs

        all_QA = dataset_format_converstion(data_path)
        all_QA_list = []
        for sublist in all_QA.values():
            all_QA_list.extend(sublist)

        # Convert list into dictionary to use Dataset.from_dict function
        QA_dict = {}
        # Loop through each key in the first item to initialize the dictionary structure
        for key in all_QA_list[0]:
            QA_dict[key] = []

        # Populate the lists for each column
        for item in all_QA_list:
            for key, value in item.items():
                QA_dict[key].append(value)
        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
                )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {"input_ids": torch.stack(pad_input_ids_list).squeeze(),
                "label": torch.stack(label_list).squeeze(),
                "attention_mask": torch.stack(pad_attention_mask_list).squeeze()}


class QAForgetDataset(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer, 
                 configs,  
                 max_length=512, 
                 split = None, 
                 loss_type=None
                 ):
        super(QAForgetDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        all_QA = dataset_format_converstion(data_path)

        forget_edge = configs.forget_edge
        forget_dict, retain_dict = {}, {}

        for key, value in all_QA.items():
            if key in forget_edge:
                forget_dict[key] = value
            else:
                retain_dict[key] = value

        forget_QA_list, retain_QA_list = [], []
        for sublist in forget_dict.values():
            forget_QA_list.extend(sublist)
        for sublist in retain_dict.values():
            retain_QA_list.extend(sublist)

        # Convert list into dictionary to use Dataset.from_dict function
        QA_dict_forget, QA_dict_retain = {}, {}
        # Loop through each key in the first item to initialize the dictionary structure
        for key in forget_QA_list[0]:
            QA_dict_forget[key] = []
        for item in forget_QA_list:
            for key, value in item.items():
                QA_dict_forget[key].append(value)
        self.forget_data = HFDataset.from_dict(QA_dict_forget)

        for key in retain_QA_list[0]:
            QA_dict_retain[key] = []
        for item in retain_QA_list:
            for key, value in item.items():
                QA_dict_retain[key].append(value)
        self.retain_data = HFDataset.from_dict(QA_dict_retain)        
        

        self.loss_type = loss_type
        self.configs = configs

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in ["forget", "retain"]:
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer'] 

            converted_forget_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
                )
            rets.append(converted_forget_data)
        return rets


class QAForgetDatasetDPO(Dataset):
    def __init__(self, 
                 data_path,
                 tokenizer, 
                 configs, 
                 max_length=512, 
                 split = None,
                 loss_type=None
                 ):
        super(QAForgetDatasetDPO, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        all_QA = dataset_format_converstion(data_path)
        
        forget_edge = configs.forget_edge
        forget_dict, retain_dict = {}, {}

        for key, value in all_QA.items():
            if key in forget_edge:
                forget_dict[key] = value
            else:
                retain_dict[key] = value

        forget_QA_list, retain_QA_list = [], []
        for sublist in forget_dict.values():
            forget_QA_list.extend(sublist)
        for sublist in retain_dict.values():
            retain_QA_list.extend(sublist)

        # Convert list into dictionary to use Dataset.from_dict function
        QA_dict_forget, QA_dict_retain = {}, {}
        # Loop through each key in the first item to initialize the dictionary structure
        for key in forget_QA_list[0]:
            QA_dict_forget[key] = []
        for item in forget_QA_list:
            for key, value in item.items():
                QA_dict_forget[key].append(value)
        self.forget_data = HFDataset.from_dict(QA_dict_forget)

        for key in retain_QA_list[0]:
            QA_dict_retain[key] = []
        for item in retain_QA_list:
            for key, value in item.items():
                QA_dict_retain[key].append(value)
        self.retain_data = HFDataset.from_dict(QA_dict_retain)       
        
        self.idontknowfile = "data/idontknow.jsonl" 
        self.idk = open(self.idontknowfile, "r").readlines()
        
        self.loss_type = loss_type
        self.configs = configs

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
            )
            rets.append(converted_data)
        return rets


class QAForgetEdgeDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 configs,
                 max_length=512, 
                 split = None, 
                 question_key='question', 
                 answer_key='answer'
                 ):
        super(QAForgetEdgeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs

        all_QA = dataset_format_converstion(data_path)
        
        forget_edge = configs.forget_edge
        forget_dict = {}

        for key, value in all_QA.items():
            if key in forget_edge:
                forget_dict[key] = value

        forget_QA_list = []
        for sublist in forget_dict.values():
            forget_QA_list.extend(sublist)

        QA_dict = {}
        for key in forget_QA_list[0]:
            QA_dict[key] = []
        # Populate the lists for each column
        for item in forget_QA_list:
            for key, value in item.items():
                QA_dict[key].append(value)
        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)

        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
                )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {"input_ids": torch.stack(pad_input_ids_list).squeeze(),
                "label": torch.stack(label_list).squeeze(),
                "attention_mask": torch.stack(pad_attention_mask_list).squeeze()}


class QARetainedEdgeDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 configs,
                 max_length=512, 
                 split = None, 
                 question_key='question', 
                 answer_key='answer'
                 ):
        super(QARetainedEdgeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs

        all_QA = dataset_format_converstion(data_path)
        
        forget_edge = configs.forget_edge
        retain_dict = {}

        for key, value in all_QA.items():
            if key not in forget_edge:
                retain_dict[key] = value

        retain_QA_list = []
        for sublist in retain_dict.values():
            retain_QA_list.extend(sublist)

        QA_dict = {}
        for key in retain_QA_list[0]:
            QA_dict[key] = []
        # Populate the lists for each column
        for item in retain_QA_list:
            for key, value in item.items():
                QA_dict[key].append(value)
        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)

        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
                )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {"input_ids": torch.stack(pad_input_ids_list).squeeze(),
                "label": torch.stack(label_list).squeeze(),
                "attention_mask": torch.stack(pad_attention_mask_list).squeeze()}


class QAIndependentSalesDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 configs,
                 max_length=512, 
                 split = None, 
                 question_key='question', 
                 answer_key='answer'
                 ):
        super(QAIndependentSalesDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs

        all_QA = dataset_format_converstion(data_path)

        independent_sales_edge = configs.independent_sales_edge
        independent_sales_edge_dict = {}

        for key, value in all_QA.items():
            if key in independent_sales_edge:
                independent_sales_edge_dict[key] = value
        
        independent_sales_edge_list = []
        for sublist in independent_sales_edge_dict.values():
            independent_sales_edge_list.extend(sublist)

        # Convert list into dictionary to use Dataset.from_dict function
        QA_dict = {}
        # Loop through each key in the first item to initialize the dictionary structure
        for key in independent_sales_edge_list[0]:
            QA_dict[key] = []

        # Populate the lists for each column
        for item in independent_sales_edge_list:
            for key, value in item.items():
                QA_dict[key].append(value)
        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)

        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
                )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {"input_ids": torch.stack(pad_input_ids_list).squeeze(),
                "label": torch.stack(label_list).squeeze(),
                "attention_mask": torch.stack(pad_attention_mask_list).squeeze()}


class QAIndependentEmploymentDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 configs,
                 max_length=512, 
                 split = None, 
                 question_key='question', 
                 answer_key='answer'
                 ):
        super(QAIndependentEmploymentDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs

        all_QA = dataset_format_converstion(data_path)
        
        indepedent_employment_edge = configs.indepedent_employment_edge
        indepedent_employment_edge_dict = {}

        for key, value in all_QA.items():
            if key in indepedent_employment_edge:
                indepedent_employment_edge_dict[key] = value

        indepedent_employment_edge_list = []
        for sublist in indepedent_employment_edge_dict.values():
            indepedent_employment_edge_list.extend(sublist)

        # Convert list into dictionary to use Dataset.from_dict function
        QA_dict = {}
        # Loop through each key in the first item to initialize the dictionary structure
        for key in indepedent_employment_edge_list[0]:
            QA_dict[key] = []

        # Populate the lists for each column
        for item in indepedent_employment_edge_list:
            for key, value in item.items():
                QA_dict[key].append(value)

        # Now, create the dataset
        self.data = HFDataset.from_dict(QA_dict)

        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
                )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {"input_ids": torch.stack(pad_input_ids_list).squeeze(),
                "label": torch.stack(label_list).squeeze(),
                "attention_mask": torch.stack(pad_attention_mask_list).squeeze()}
    

class QAFactualDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 configs,
                 max_length=512, 
                 split = None, 
                 question_key='question', 
                 answer_key='answer'
                 ):
        super(QAFactualDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.configs = configs
        self.data = load_dataset('json', data_files = data_path, split=split)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_qa(
                self.tokenizer, self.max_length, question, answer, self.configs
                )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return {"input_ids": torch.stack(pad_input_ids_list).squeeze(),
                "label": torch.stack(label_list).squeeze(),
                "attention_mask": torch.stack(pad_attention_mask_list).squeeze()}
    

def custom_data_collator(samples):
    input_ids = [s['input_ids'] for s in samples]
    labels = [s['label'] for s in samples]
    attention_mask = [s['attention_mask'] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets


def custom_data_collator_forget_dpo(samples):
    idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
    rets = []
    for data_type in ["idk", "forget", "retain"]:
        if data_type == "idk":
            data = idk_samples
        elif data_type == "forget":
            data = forget_samples
        else:
            data = retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss