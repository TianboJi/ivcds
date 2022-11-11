modelname="gpt2"
savename="gpt2"
dataset="~/ivcds/data/data_for_clm"

python run_clm.py \
    --model_name_or_path $modelname \
    --dataset_dir $dataset \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 40 \
    --do_train \
    --do_eval \
    --learning_rate=5e-5 \
    --save_steps 200 \
    --output_dir "clm-output/"$savename \
    --overwrite_output_dir \
    --save_total_limit 20 \
    --preprocessing_num_workers 10 \
    --dataloader_num_workers 10 \
