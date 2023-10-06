export MS_ASCEND_CHECK_OVERFLOW_MODE=1 # for ms+910B, check overflow

python infer.py\
    --cfg configs/t2v.yaml\
    --seed 9999\
    --input_video "datasets/webvid5/1.mp4"\
    --input_text_desc "Disco light leaks disco ball light reflections shaped rectangular and line with motion blur effect." \
    --resume_checkpoint $1 


