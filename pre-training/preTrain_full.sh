store_path="./"
dataset="reddit"
small="0"
negative="1"
force_accuracy="0.5"
reinit="2"
init_rounds="3"
stage_two_trunk="1"
min_mstep_rounds="2"
patience="1000"
prefix="link_fast"
trunk_size="600000"
trunk_em_rounds="3"
mlm_weight="0.5"
link_weight="1.0"
n_hop="3"

model_size="base"
model_type="bert"
model_name="bert-${model_size}-uncased"
batch_size="64"
learning_rate="2e-5"
keep_layers="12"

# model_size="large"
# model_type="electra"
# model_name="google/electra-${model_size}-discriminator"
# batch_size="16"
# learning_rate="8e-6"
# keep_layers="24"

cuda="0"
addr="0"
cpu_count="64"
model_file="embindicator_pooler"
init_path="last_utterance"

TOKENIZERS_PARALLELISM=false accelerate launch --config_file accelerator_config.yml acc_preTrain_full.py \
    --save_path ${prefix}_save \
    --small $small \
    --dataset ${dataset} \
    --cpu_count $cpu_count \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --model_file $model_file \
    --init_path $init_path \
    --addr $addr \
    --negative $negative \
    --keep_layers $keep_layers \
    --trunk_size $trunk_size \
    --stage_two_trunk $stage_two_trunk \
    --trunk_em_rounds $trunk_em_rounds \
    --mlm_weight $mlm_weight \
    --link_weight $link_weight \
    --n_hop $n_hop \
    --force_accuracy $force_accuracy \
    --reinit $reinit \
    --init_rounds $init_rounds \
    --min_mstep_rounds $min_mstep_rounds \
    --patience $patience \
    --model_type $model_type \
    --model_name $model_name \
    --store_path $store_path
