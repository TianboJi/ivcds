modelname="gpt2"
savename="gpt2"
mode="history-profile"
dataset=/home/jitianbo/Workspace/driver_simulator_kvret/data/data_for_clm_ablation/$mode
num_epochs=50

python run_clm.py \
    --model_name_or_path $modelname \
    --dataset_dir $dataset \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs $num_epochs \
    --do_train \
    --do_eval \
    --learning_rate=5e-5 \
    --save_steps 1000 \
    --output_dir ablation-output/$mode-$savename \
    --overwrite_output_dir \
    --save_total_limit 3 \
    --preprocessing_num_workers 8 \

