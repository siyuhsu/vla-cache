
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
    --task_suite_name libero_object    \
    --use_vla_cache False \