
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
    --task_suite_name libero_10    \
    --use_vla_cache False \