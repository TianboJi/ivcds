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
    
    model.eval()

    save_dir = Path(inference_args.output_dir).absolute().resolve()
    save_dir.mkdir(exist_ok=True)
    model_dir = save_dir.joinpath(f"{inference_args.model_name}")
    model_dir.mkdir(exist_ok=True)
    
    nlu_raw_fpath = model_dir.joinpath(f"{dset}-NLU.raw")
    nlu_result_fpath = model_dir.joinpath(f"{dset}-NLU.result")
    nlu_target_fpath = model_dir.joinpath(f"{dset}-NLU.target")
    
    pol_raw_fpath = model_dir.joinpath(f"{dset}-POL.raw")
    pol_result_fpath = model_dir.joinpath(f"{dset}-POL.result")
    pol_target_fpath = model_dir.joinpath(f"{dset}-POL.target")

    nlg_raw_fpath = model_dir.joinpath(f"{dset}-NLG.raw")
    nlg_result_fpath = model_dir.joinpath(f"{dset}-NLG.result")
    nlg_target_fpath = model_dir.joinpath(f"{dset}-NLG.target")

    with nlu_raw_fpath.open('w') as f_nlu_raw, \
         nlu_result_fpath.open('w') as f_nlu_result, \
         nlu_target_fpath.open('w') as f_nlu_target, \
         pol_raw_fpath.open('w') as f_pol_raw, \
         pol_result_fpath.open('w') as f_pol_result, \
         pol_target_fpath.open('w') as f_pol_target, \
         nlg_raw_fpath.open('w') as f_nlg_raw, \
         nlg_result_fpath.open('w') as f_nlg_result, \
         nlg_target_fpath.open('w') as f_nlg_target, \
         torch.no_grad():

        for source_text,raw_target in tqdm(zip(source_data,target_data),total=len(source_data)):

            input_text = source_text
            input_ids = tokenizer(input_text, return_tensors="pt").to(device).input_ids
            input_len = input_ids.shape[-1]
            max_len = 80

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
            # opt = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
            generated = outputs[:,input_len:]
            # to_decode = decode_by_task(generated,tokenizer,task)
            # results = tokenizer.batch_decode(to_decode, skip_special_tokens=False,clean_up_tokenization_spaces=False)
            results = tokenizer.batch_decode(generated, skip_special_tokens=False,clean_up_tokenization_spaces=True)
            raw_result = results[0].strip()
            raw_result = raw_result.replace("][","] [")

            result = process_raw_result(raw_result,task)

            # raw_target = target_data[i]
            target = remove_special_sep_tokens(raw_target)

            if task == "NLU":
                f_nlu_raw.write(f"{raw_result}\n")
                f_nlu_result.write(f"{result}\n")
                f_nlu_target.write(f"{target}\n")
                f_nlu_raw.flush()
                f_nlu_result.flush()
                f_nlu_target.flush()
            elif task == "POL":
                f_pol_raw.write(f"{raw_result}\n")
                f_pol_result.write(f"{result}\n")
                f_pol_target.write(f"{target}\n")
                f_pol_raw.flush()
                f_pol_result.flush()
                f_pol_target.flush()
            elif task == "NLG":
                f_nlg_raw.write(f"{raw_result}\n")
                f_nlg_result.write(f"{result}\n")
                f_nlg_target.write(f"{target}\n")
                f_nlg_raw.flush()
                f_nlg_result.flush()
                f_nlg_target.flush()





    
if __name__ == "__main__":
    main()