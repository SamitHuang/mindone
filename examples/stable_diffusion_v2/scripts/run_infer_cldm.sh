export DEVICE_ID=$1

{"source": "source/39860.png", "target": "target/39860.png", "prompt": "coral circle with blue background"}
{"source": "source/2274.png", "target": "target/2274.png", "prompt": "medium sea green circle with black background"}
{"source": "source/3341.png", "target": "target/3341.png", "prompt": "light green circle with snow background"}
{"source": "source/40455.png", "target": "target/40455.png", "prompt": "dark turquoise circle with medium spring green background"}

task_name=$3
scale=9

cd inference

output_dir=output/${task_name}_1
rm -rf $output_dir
mkdir -p $output_dir

python sd_infer.py \
--device_target=Ascend \
--task=controlnet \
--model=./config/model/v1-inference-controlnet.yaml \
--sampler=./config/schedule/ddim.yaml \
--sampling_steps=50 \
--scale=$scale \
--n_iter=2 \
--n_samples=4 \
--controlnet_mode=canny \
--control_path "../datasets/fill50k/source/39860.png" \
--image_path "../datasets/fill50k/target/39860.png" \
--prompt "coral circle with blue background" \
--a_prompt "" \
--negative_prompt "" \
--pretrained_ckpt=../$2 \
--output_path=$output_dir \
# > $output_dir/infer.log 2>&1 &


output_dir=output/${task_name}_2
rm -rf $output_dir
mkdir -p $output_dir
python sd_infer.py \
--device_target=Ascend \
--task=controlnet \
--model=./config/model/v1-inference-controlnet.yaml \
--sampler=./config/schedule/ddim.yaml \
--sampling_steps=50 \
--scale=$scale \
--n_iter=2 \
--n_samples=4 \
--controlnet_mode=canny \
--control_path "../datasets/fill50k/source/2274.png" \
--image_path "../datasets/fill50k/target/2274.png" \
--prompt "medium sea green circle with black background" \
--a_prompt "" \
--negative_prompt "" \
--pretrained_ckpt=../$2 \
--output_path=$output_dir \
# > $output_dir/infer.log 2>&1 &


output_dir=output/${task_name}_3
rm -rf $output_dir
mkdir -p $output_dir
python sd_infer.py \
--device_target=Ascend \
--task=controlnet \
--model=./config/model/v1-inference-controlnet.yaml \
--sampler=./config/schedule/ddim.yaml \
--sampling_steps=50 \
--scale=$scale \
--n_iter=2 \
--n_samples=4 \
--controlnet_mode=canny \
--control_path "../datasets/fill50k/source/3341.png" \
--image_path "../datasets/fill50k/target/3341.png" \
--prompt "light green circle with snow background" \
--a_prompt "" \
--negative_prompt "" \
--pretrained_ckpt=../$2 \
--output_path=$output_dir \
# > $output_dir/infer.log 2>&1 &


output_dir=output/${task_name}_4
rm -rf $output_dir
mkdir -p $output_dir
python sd_infer.py \
--device_target=Ascend \
--task=controlnet \
--model=./config/model/v1-inference-controlnet.yaml \
--sampler=./config/schedule/ddim.yaml \
--sampling_steps=50 \
--scale=$scale \
--n_iter=2 \
--n_samples=4 \
--controlnet_mode=canny \
--control_path "../datasets/fill50k/source/40455.png" \
--image_path "../datasets/fill50k/target/40455.png" \
--prompt "dark turquoise circle with medium spring green background" \
--a_prompt "" \
--negative_prompt "" \
--pretrained_ckpt=../$2 \
--output_path=$output_dir \
# > $output_dir/infer.log 2>&1 &


# in distribution
<<comment
python sd_infer.py \
--device_target=Ascend \
--task=controlnet \
--model=./config/model/v1-inference-controlnet.yaml \
--sampler=./config/schedule/ddim.yaml \
--sampling_steps=50 \
--scale=$scale \
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
comment

echo "All results will be saved under ./inference/${output_dir}"
#--pretrained_ckpt=../output/train_cldm_canny_fill50k/ckpt/sd-16.ckpt \
#--pretrained_ckpt="../models/control_sd15_canny_ms_static.ckpt" \
