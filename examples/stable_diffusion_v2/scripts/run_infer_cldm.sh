export DEVICE_ID=$1

task_name=infer_cldm
output_dir=output/$task_name

cd inference
mkdir -p $output_dir

python sd_infer.py \
--device_target=Ascend \
--task=controlnet \
--model=./config/model/v1-inference-controlnet.yaml \
--sampler=./config/schedule/ddim.yaml \
--sampling_steps=50 \
--n_iter=2 \
--n_samples=4 \
--controlnet_mode=canny \
--control_path "../datasets/fill5/source/0.png" \
--image_path "../datasets/fill5/target/0.png" \
--prompt "pale golden rod circle with old lace background" \
--a_prompt "" \
--negative_prompt "" \
--pretrained_ckpt=../$2 \
--output_path=$output_dir \


echo "All results will be saved under ./inference folder"
#--pretrained_ckpt=../output/train_cldm_canny_fill50k/ckpt/sd-16.ckpt \
#--pretrained_ckpt="../models/control_sd15_canny_ms_static.ckpt" \
