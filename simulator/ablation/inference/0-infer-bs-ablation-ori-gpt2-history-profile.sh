modelname="gpt2"
mode="history-profile"
modeldir=~/ivcds/simulator/clm-output/$modelname
dataset_dir=~/ivcds/data/data_for_clm_ablation/$mode
output_dir=~/ivcds/simulator/ablation/inference/inference-results

python inference_clm_bs.py \
    --model_path $modeldir \
    --model_name ori-$mode-$modelname-bs \
    --batch_size 64 \
    --dataset_dir $dataset_dir \
    --dset test \
    --device cuda \
    --output_dir $output_dir 
