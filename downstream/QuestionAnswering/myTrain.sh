store_path="./"
dataset="molweni"

epochs="5"
model_file="embindicator"

model_size="base"
batch_size="16"
learning_rate="7e-5"
model_type="bert"
model_name="bert-base-uncased"

# model_size="large"
# batch_size="8"
# learning_rate="4e-5"
# model_type="electra"
# model_name="google/electra-${model_size}-discriminator"

# pretrain_name="None"
pretrain_name="bert-our"
# pretrain_name="electra-our"
pretrain_path="pretrain_models/${pretrain_name}.pth"


python3 myTrain.py \
    --cuda $1 \
    --dataset $dataset \
    --epochs $epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --model_file $model_file \
    --model_type $model_type \
    --model_name $model_name \
    --pretrain_path $pretrain_path \
    --store_path $store_path \
    --save_path ${pretrain_name}_save
