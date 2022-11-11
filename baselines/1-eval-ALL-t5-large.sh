savename="t5-large"
evalset="test"
batchsize=4

task="NLU"
ckp="~/ivcds/baselines/trained_models/NLU-t5-large/"
echo "Model '"$ckp"' evaluating on task '"$task"' with '"$evalset"' set"
python run_eval.py $ckp $task/$evalset.source  predictions/$task-$savename-$evalset.txt \
    --reference_path $task/$evalset.target \
    --score_path enro_bleu_short-long.json \
    --task summarization \
    --device cuda \
    --bs $batchsize

task="POL"
ckp="~/ivcds/baselines/trained_models/POL-t5-large/"
echo "Model '"$ckp"' evaluating on task '"$task"' with '"$evalset"' set"
python run_eval.py $ckp $task/$evalset.source  predictions/$task-$savename-$evalset.txt \
    --reference_path $task/$evalset.target \
    --score_path enro_bleu_short-long.json \
    --task summarization \
    --device cuda \
    --bs $batchsize

task="NLG"
ckp="~/ivcds/baselines/trained_models/NLG-t5-large/"
echo "Model '"$ckp"' evaluating on task '"$task"' with '"$evalset"' set"
python run_eval.py $ckp $task/$evalset.source  predictions/$task-$savename-$evalset.txt \
    --reference_path $task/$evalset.target \
    --score_path enro_bleu_short-long.json \
    --task summarization \
    --device cuda \
    --bs $batchsize

