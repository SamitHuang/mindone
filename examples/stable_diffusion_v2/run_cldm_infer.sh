export DEVICE_ID=$1

cd inference
python sd_infer.py \
--device_target=Ascend \
--task=controlnet \
--model=./config/model/v1-inference-controlnet.yaml \
--sampler=./config/schedule/ddim.yaml \
--sampling_steps=50 \
--n_iter=2 \
--n_samples=4 \
--controlnet_mode=canny \
--control_path "../datasets/fill5/source/3.png" \
--image_path "../datasets/fill5/target/3.png" \
--prompt "cornflower blue circle with light golden rod yellow background" \
--a_prompt "" \
--negative_prompt "" \
--pretrained_ckpt=../output/train_cldm_canny_fill50k_moreFp32_wd1e-3/ckpt/sd-59.ckpt \
--output_path=../output/infer_cldm_fill50k/sd15_controlnet_tr3e59i3 \

#--pretrained_ckpt=../output/train_cldm_canny_fill50k/ckpt/sd-16.ckpt \
#--output_path=../output/infer_cldm_fill50k/sd15_controlnet_tr2e16i3 \



#--pretrained_ckpt=../models/sd_v1.5-d0ab7146_controlnet_init.ckpt \
#--output_path=../output/infer_cldm_fill5/sd15_controlnet_init
#"../output/cldm_canny/ckpt/sd-5.ckpt" \
#--pretrained_ckpt="../models/control_sd15_canny_ms_static.ckpt" \


# --image_path "../test_imgs/dog2.png" \
# --prompt "cute dog" \

