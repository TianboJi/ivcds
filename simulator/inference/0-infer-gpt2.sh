modeldir='/home/jitianbo/Workspace/driver_simulator_kvret/simulator/clm-output/gpt2/'
modelname="gpt2"
dataset_dir="/home/jitianbo/Workspace/driver_simulator_kvret/data/data_for_clm/"
output_dir="/home/jitianbo/Workspace/driver_simulator_kvret/simulator/inference/inference-results"

python inference_clm.py \
    --model_path $modeldir \
    --model_name $modelname \
    --dataset_dir $dataset_dir \
    --dset test \
    --device cuda \
    --output_dir $output_dir \
