store_path="./"
dataset="molweni"

model_size="base"
batch_size="32"
learning_rate="1e-4"
model_type="bert"
model_name="bert-base-uncased"

# model_size="large"
# batch_size="12"
# learning_rate="5e-5"
# model_type="electra"
# model_name="google/electra-${model_size}-discriminator"

# pretrain_name="None"
pretrain_name="bert-our"
# pretrain_name="electra-our"

pretrain_path="pretrain_models/${pretrain_name}.pth"

epochs="30"
logging_times="10"
cpu_count="64"
model_file="embindicator"

accelerate launch --config_file accelerator_config.yml myTrain.py \
    --dataset $dataset \
    --cpu_count $cpu_count \
    --epochs $epochs \
    --logging_times $logging_times \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --model_file $model_file \
    --model_type $model_type \
    --model_name $model_name \
    --pretrain_path $pretrain_path \
    --store_path $store_path \
    --save_path ${pretrain_name}_save
