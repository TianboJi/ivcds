# IvCDS: An End-to-end Driver Simulator For Personal In-vehicle Conversational Assistant

## Structure
- Files for IvCDS are stored in directory `simulator`, including the ablation study.
- Files for baselines are stored in directory `baselines`
- Directory `data` contains the processed KVRET dataset



## Environment
- Python=3.8
- PyTorch
- Transformers=4.18.0

## IvCDS
- Files for training, inference and ablation study are available  in directory `simulator`. 
- `run_clm.py` is the file for training, while `0-train-gpt-2.sh` is the running script.
- `inference/0-infer-gpt2.sh` is the script for interence.
- `inference/eval.ipynb` is a jupyter file for the evaluation after IvCDS's training&interence.
- Directory `ablation` includes all scripts for ablation study. We provide three example scripts, please check any of them. 



## Baseslines
- Files For baselines is under directory `baselines`
- We use T5-large as an example, where `0-train-ALL-t5-large.sh` and `1-eval-ALL-t5-large.sh` are the files for training and inference, respectively.
- All baseline models in this paper are pretrained, and they can be found on https://huggingface.co/models

## Notes
1. Before running a bash script, please modify the paths in it to the paths of your own enviroment.
2. The training of IvCDS takes about two hours on a single RTX 3090. 
3. Due to the file size, we cannot include the trained checkpoints. If you would like to use them, please contact us.
4. The paper is still under review and this repository is at an early stage. We may update it in the future
5. If you encounter any problem, please email jitianbo@gmail.com

