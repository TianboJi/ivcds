modelname="t5-large"
savename="t5-large"
evalset="test"
batchsize=8

task="NLU"

echo "Model '"$modelname"' training on task '"$task"'"
python finetune_trainer_sp.py \
    --model_name_or_path $modelname \
    --data_dir $task/ \
    --task summarization \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size $batchsize \
    --per_device_eval_batch_size $batchsize \
    --output_dir trained_models/$task-$savename/ \
    --learning_rate=1e-5 \
    --num_train_epochs 20 \
    --do_train \
    --do_eval \
    --n_val 1000 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --overwrite_output_dir \


task="POL"

echo "Model '"$modelname"' training on task '"$task"'"
python finetune_trainer_sp.py \
    --model_name_or_path $modelname \
    --data_dir $task/ \
    --task summarization \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size $batchsize \
    --per_device_eval_batch_size $batchsize \
    --output_dir trained_models/$task-$savename/ \
    --learning_rate=1e-5 \
    --num_train_epochs 20 \
    --do_train \
    --do_eval \
    --n_val 1000 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --overwrite_output_dir \


task="NLG"

echo "Model '"$modelname"' training on task '"$task"'"
python finetune_trainer_sp.py \
    --model_name_or_path $modelname \
    --data_dir $task/ \
    --task summarization \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size $batchsize \
    --per_device_eval_batch_size $batchsize \
    --output_dir trained_models/$task-$savename/ \
    --learning_rate=1e-5 \
    --num_train_epochs 20 \
    --do_train \
    --do_eval \
    --n_val 1000 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --overwrite_output_dir \


