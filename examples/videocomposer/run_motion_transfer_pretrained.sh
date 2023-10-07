# Exp02, Motion Transfer from a video to a Single Image
<<com
python infer.py \
    --cfg configs/exp02_motion_transfer.yaml \
    --seed 9999 \
    --input_video "demo_video/motion_transfer.mp4" \
    --image_path "demo_video/sunflower.png" \
    --input_text_desc "A sunflower in a field of flowers" \
    --ms_mode 0
com

python infer.py\
    --cfg configs/exp02_motion_transfer.yaml\
    --seed 9999\
    --input_video "datasets/webvid5/2.mp4"\
    --image_path "datasets/webvid5/vid2_frm1.png"\
    --input_text_desc "Cloudy moscow kremlin time lapse" \
    --ms_mode 1


