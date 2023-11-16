export DEVICE_ID=$1

task_name=$3 #infer_cldm_pretrained
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
--pretrained_ckpt=$2 \
--output_path=$output_dir \

#--pretrained_ckpt=../output/train_cldm_canny_fill50k/ckpt/sd-16.ckpt \
#--output_path=../output/infer_cldm_fill50k/sd15_controlnet_tr2e16i3 \



#--pretrained_ckpt=../models/sd_v1.5-d0ab7146_controlnet_init.ckpt \
#--output_path=../output/infer_cldm_fill5/sd15_controlnet_init
#"../output/cldm_canny/ckpt/sd-5.ckpt" \
#--pretrained_ckpt="../models/control_sd15_canny_ms_static.ckpt" \


# --image_path "../test_imgs/dog2.png" \
# --prompt "cute dog" \
