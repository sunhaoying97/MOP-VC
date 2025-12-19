num_runs=$1
gpu=$2

# by default, we run 5 times for every experiment
num_runs=${num_runs:-5}
gpu=${gpu:-0}


base_cmd="python train.py \
--dataset MSRVTT \
--arch base \
--method PairTransformer \
--modality ami \
--decoder_modality_flags VA \
--predictor_modality_flags VA \
--batch_size 512 \
--num_workers 6"


## Baseline
cmd="$base_cmd --task Class --feats ViT"
bash scripts/run.sh "$cmd" $num_runs $gpu
