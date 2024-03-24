# Parallel config
# Please generate the rank table file via hccl_tools.py
# (https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) for your own server
# num_devices=4
# rank_table_file=/home/hyx/tools/hccl_4p_4567_127.0.0.1.json
# CANDIDATE_DEVICE=(4 5 6 7)

num_devices=2
rank_table_file=/home/hyx/tools/hccl_2p_67_127.0.0.1.json
CANDIDATE_DEVICE=(6 7)


export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
export RANK_TABLE_FILE=$rank_table_file
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

# Training path config
output_path=outputs
task_name=vae_train_2p

# parallel train
rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
cp $0 $output_path/.

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0; i<${RANK_SIZE}; i++))
do
    export RANK_ID=$((rank_start + i))
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    mkdir -p ${output_path:?}/${task_name:?}/rank_$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    python train_autoencoder.py --base_config configs/train/autoencoder_kl_f8.yaml \
        --mode=0 \
        --csv_path="datasets/chinese_art_blip/train/img_txt.csv" \
        --data_path="datasets/chinese_art_blip/train" \
        --dtype="fp32" \
        --init_loss_scale=1 \
        --epochs=1000 \
        --ckpt_save_interval=100 \
        --batch_size=4 \
        --flip=True \
        --random_crop=True \
        --use_parallel=True \
        --output_path=$output_path/$task_name \
        > $output_path/$task_name/rank_$i/train.log 2>&1 &
done
