# IvCDS: An End-to-end Driver Simulator For Personal In-vehicle Conversational Assistant
This repository is the code for our paper (see this [link](https://www.mdpi.com/1660-4601/19/23/15493)). 

## Citation (BibTeX)
If you are interested in our work, please cite it by:
```
@Article{ijerph192315493,
    AUTHOR = {Ji, Tianbo and Yin, Xuanhua and Cheng, Peng and Zhou, Liting and Liu, Siyou and Bao, Wei and Lyu, Chenyang},
    TITLE = {IvCDS: An End-to-End Driver Simulator for Personal In-Vehicle Conversational Assistant},
    JOURNAL = {International Journal of Environmental Research and Public Health},
    VOLUME = {19},
    YEAR = {2022},
    NUMBER = {23},
    ARTICLE-NUMBER = {15493},
    URL = {https://www.mdpi.com/1660-4601/19/23/15493},
    ISSN = {1660-4601},
    DOI = {10.3390/ijerph192315493}
}
```

## File Structure
- Files for IvCDS are stored in directory `simulator/`, including the ablation study.

- Files for baselines are stored in directory `baselines/`
- Directory `data/` contains the processed KVRET dataset



## Requirement
+ Python=3.8
+ Pytorch
+ Transformers=4.18.0

## Our model: IvCDS
- Files for training, inference and ablation study are available in directory `simulator/`. 

- `run_clm.py` is the file for training, while `0-train-gpt-2.sh` is the running script.
- `inference/0-infer-gpt2.sh` is the script for inference.
- `inference/eval.ipynb` is a jupyter file for the evaluation after IvCDS's training&inference.
- Directory `ablation/` includes all scripts for ablation study. We provide three example scripts, please check any of them. 



## Baseline models
- Files For baselines is under directory `baselines/`.

- We use T5-large as an example, where `0-train-ALL-t5-large.sh` and `1-eval-ALL-t5-large.sh` are the files for training and inference, respectively.
- The evaluation of baselines uses the exact same file as IvCDS, see `inference/eval.ipynb`.
- All baseline models in this paper are pretrained, and they can be found on [HuggingFace Models](https://huggingface.co/models).



## Notes
1. Before running a bash script, please modify the paths in it to the paths of your own environment.
2. The training of IvCDS takes about two hours on a single RTX 3090. 
3. Checkpoints for models in our paper are available in this [link](https://drive.google.com/drive/folders/1xZYvE3sX59aOgB_9bj_yt5SnfNtzFcgV?usp=sharing).
4. The paper is still under review and this repository is at an early stage. The instructions may still lack details, and we will update it in the future.
5. If you encounter any problem, please email me by jitianbo@gmail.com.

