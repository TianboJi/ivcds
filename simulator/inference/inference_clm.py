import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import os,json
from dataclasses import dataclass,field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    MODEL_FOR_CAUSAL_LM_MAPPING,
)
import torch
from torch.utils.data.dataset import Dataset
import datasets
from datasets import load_dataset, load_metric

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class InferenceArguments:
    """
    Arguments of model/config/tokenizer/dataset for inference
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint."},
    )

    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Model name"},
    )
    
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "directory to dataset"},
    )
    
    dset: Optional[str] = field(
        default="test",
        metadata={"help": "dataset type: train, dev, test"},
    )
    device: Optional[str] = field(
        default="cpu",
        metadata={"help": "Model name"},
    )
    
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "output_dir"},
    )

    

def load_model(modelpath, device='cuda'):
    # config = AutoConfig.from_pretrained(modelname)
    model = AutoModelForCausalLM.from_pretrained(modelpath).to(device)
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    return model, tokenizer

def detect_task_by_source(source_text):
    # NUL task: assistant utterance -> assistant action
    if source_text.endswith('[eoau]'): 
        return "NLU"
    # POL task: assistant action -> driver action
    elif source_text.endswith('[eoaa]'): 
        return "POL"
    # NLG task: driver action -> driver utterance
    else:
        return "NLG"
def get_eos_by_task(task, tokenizer):
    eos_dict = {
        "NLU": "[eoaa]",
        "POL": "[eoda]",
        "NLG": "[eodu]",
    }
    return tokenizer.encode(eos_dict[task])[0]

def get_sos_by_task(task, tokenizer):
    sos_dict = {
        "NLU": "[soaa]",
        "POL": "[soda]",
        "NLG": "[sodu]",
    }
    return tokenizer.encode(sos_dict[task])[0]


def process_generated_text(result, task):
    words = result.split()
    eos_dict = {
        "NLU": "[eoaa]",
        "POL": "[eoda]",
        "NLG": "[eodu]",
    }
    sos_dict = {
        "NLU": "[soaa]",
        "POL": "[soda]",
        "NLG": "[sodu]",
    }
    sos_token = sos_dict[task]
    eos_token = eos_dict[task]
    if sos_token not in words or eos_token not in words:
        return result
    sos_id = words.index(sos_token)
    eos_id = words.index(eos_token)
    if sos_id < eos_id:
        tokens = words[sos_id:eos_id+1]
        return ' '.join(tokens)
    words_np = np.array(words)
    # sos_ids,*_ = np.where(words_np==sos_token)
    eos_ids,*_ = np.where(words_np==eos_token)
    
    # 如果sos的第一个id比eos最后一个id还要大，说明不存在valid的数据，原样返回
    if sos_id > eos_ids[-1]:
        return result
    
    for e in eos_ids:
        if e > sos_id:
            break
    eos_id = e
    tokens = words[sos_id:eos_id+1]
    return ' '.join(tokens)

def remove_special_sep_tokens(text):
    special_sep_tokens = [
        '[eoaa]', '[eoau]', '[eoda]', '[eodp]', '[eodu]',
        '[soaa]', '[soau]', '[soda]', '[sodp]', '[sodu]',
    ]
    text = text.replace("][","] [")
    words = text.split()
    tokens = [e for e in words if e not in special_sep_tokens]
    tokens = [e for e in tokens if not (e.startswith("[") and not e.endswith("]"))]
    return ' '.join(tokens)

def process_raw_result(raw_result, task):
    return remove_special_sep_tokens(process_generated_text(raw_result, task))

# def inference()

def main():
    parser = HfArgumentParser((InferenceArguments, ))
    inference_args,*_ = parser.parse_args_into_dataclasses()
    print(inference_args)
    dataset_dir = inference_args.dataset_dir
    dset = inference_args.dset
    dataset_dirpath = Path(dataset_dir).absolute().resolve()
    
    source_fpath = dataset_dirpath.joinpath(f"{dset}.source")
    with source_fpath.open() as f:
        source_data = f.read().strip().splitlines()
       
    target_fpath = dataset_dirpath.joinpath(f"{dset}.target")
    with target_fpath.open() as f:
        target_data = f.read().strip().splitlines()
    
    
    modelpath = Path(inference_args.model_path).absolute().resolve()
    # .joinpath(inference_args.model_name)
    device = inference_args.device
    model, tokenizer = load_model(
        str(modelpath),
        device=device,
    )
    # assert 1==2
    raw_result_dict = {
        "NLU": [],
        "NLG": [],
        "POL": [],
    }
    result_dict = {
        "NLU": [],
        "NLG": [],
        "POL": [],
    }
    target_dict = {
        "NLU": [],
        "NLG": [],
        "POL": [],
    }
    
    for i,source_text in enumerate(tqdm(source_data)):
        input_text = source_text

        input_ids = tokenizer(input_text, return_tensors="pt").to(device).input_ids
        input_len = input_ids.shape[-1]
        max_len = 80
        # # NUL task: assistant utterance -> assistant action

        # if source_text.endswith('[eoau]'): 
        #     eos_token_id = tokenizer.encode('[eoaa]')[0]
        # # POL task: assistant action -> driver action
        # elif source_text.endswith('[eoaa]'): 
        #     eos_token_id = tokenizer.encode('[eoda]')[0]
        # # NLG task: driver action -> driver utterance
        # elif source_text.endswith('[eoda]'): 
        #     eos_token_id = tokenizer.encode('[eodu]')[0]

        task = detect_task_by_source(source_text)
        eos_token_id = get_eos_by_task(task,tokenizer)

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
        elif  tokenizer.eos_token_id is None:
            pad_token_id = eos_token_id
        else:
            pad_token_id = tokenizer.pad_token_id
        outputs = model.generate(
            input_ids, 
            max_length=input_len+max_len,
            temperature=0.7,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        opt = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        generated = outputs[:,input_len:]
        # to_decode = decode_by_task(generated,tokenizer,task)
        # results = tokenizer.batch_decode(to_decode, skip_special_tokens=False,clean_up_tokenization_spaces=False)
        results = tokenizer.batch_decode(generated, skip_special_tokens=False,clean_up_tokenization_spaces=True)
        raw_result = results[0].strip()
        raw_result_dict[task].append(raw_result)

        result = process_raw_result(raw_result,task)
        result_dict[task].append(result)

        raw_target = target_data[i]
        target = remove_special_sep_tokens(raw_target)
        target_dict[task].append(target)

        
        
    
    save_dir = Path(inference_args.output_dir).absolute().resolve()
    save_dir.mkdir(exist_ok=True)
    model_dir = save_dir.joinpath(f"{inference_args.model_name}")
    model_dir.mkdir(exist_ok=True)
    for t in ["NLU","POL","NLG"]:
        raw_result_t = raw_result_dict[t]
        savefname = f"{dset}-{t}.raw"
        savefpath = model_dir.joinpath(savefname)
        with savefpath.open('w') as f:
            f.writelines([f"{e}\n" for e in raw_result_t])


        result_t = result_dict[t]
        savefname = f"{dset}-{t}.result"
        savefpath = model_dir.joinpath(savefname)
        with savefpath.open('w') as f:
            f.writelines([f"{e}\n" for e in result_t])

        target_t = target_dict[t]
        savefname = f"{dset}-{t}.target"
        savefpath = model_dir.joinpath(savefname)
        with savefpath.open('w') as f:
            f.writelines([f"{e}\n" for e in target_t])

    
if __name__ == "__main__":
    main()