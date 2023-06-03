store_path="./"

epochs="15"
model_file="embindicator"
filter_simple="-1.0"

learning_rate="6.25e-5"
batch_size="32"
model_size="base"
encoder_model_type="bert"
model_name="bert-base-uncased"
decoder_model_type="bert"
decoder_model_name="bert-base-uncased"

# pretrain_name="None"
pretrain_name="bert-our"
# pretrain_name="electra-our"

pretrain_path="pretrain_models/${pretrain_name}.pth"

accelerate launch --config_file accelerator_config.yml acc_myTrain.py \
    --epochs $epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --model_file $model_file \
    --filter_simple $filter_simple \
    --encoder_model_type $encoder_model_type \
    --decoder_model_type $decoder_model_type \
    --model_name $model_name \
    --decoder_model_name $decoder_model_name \
    --pretrain_path $pretrain_path \
    --store_path $store_path
