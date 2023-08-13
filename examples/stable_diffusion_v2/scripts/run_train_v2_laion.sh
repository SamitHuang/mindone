# Finetune SD2.0-base on LAION subsets

# ascend config
export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=6000
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# parallel config
use_parallel=True
num_devices=4
rank_table_file='/home/yx/tools/hccl_4p_4567_127.0.0.1.json' # regenerate the table file for your local env.
CANDIDATE_DEVICE=(4 5 6 7)

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
RANK_TABLE_FILE=$rank_table_file
export RANK_TABLE_FILE
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

# train config
data_path=/data3/datasets/laion2b_en
output_path=output/sd2.1_base
ckpt_save_interval=1

task_name=txt2img
pretrained_model_path=models/
pretrained_model_file=sd_v2_base-57526ee4.ckpt
train_config_file=configs/train_config_v2.json

# For phase 1 in training sd 2.0-base, images with resolution < 256x256 are filtered out. For phase 2, 512x512
# For sd2.1-base, images with resolution < 512x512 are filtered out
image_filter_size=512
image_size=512

train_batch_size=3
start_learning_rate=1e-5 # TODO: lr=1e-5, global bs=2048 ref from hf sd2.1 base, but lr=1e-4 in mosaicml/diffusion
end_learning_rate=5e-6  # hf: Learning rate: warmup to 0.0001 for 10,000 steps and then kept constant


warmup_steps=10000 # TDOO: related to # devices and batch size
epochs=2        # TODO: reduce for larger laion dataset
use_ema=True
clip_grad=False # TODO: confirm
max_grad_norm=1.0


# train
rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
#export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0; i<${RANK_SIZE}; i++))
do
    export RANK_ID=$((rank_start + i))
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    mkdir -p ${output_path:?}/${task_name:?}/rank_$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    nohup python -u train_text_to_image.py \
        --data_path=$data_path \
        --image_filter_size=$image_filter_size \
        --train_config=$train_config_file \
        --output_path=$output_path/$task_name \
        --use_parallel=$use_parallel \
        --pretrained_model_path=$pretrained_model_path \
        --pretrained_model_file=$pretrained_model_file \
        --image_size=$image_size \
        --train_batch_size=$train_batch_size \
        --epochs=$epochs \
        --start_learning_rate=$start_learning_rate \
        --end_learning_rate=$end_learning_rate \
        --warmup_steps=$warmup_steps \
        --ckpt_save_interval=$ckpt_save_interval\
        --use_ema=$use_ema \
        --clip_grad=$clip_grad \
        --max_grad_norm=$max_grad_norm \
        > $output_path/$task_name/rank_$i/train.log 2>&1 &
done
