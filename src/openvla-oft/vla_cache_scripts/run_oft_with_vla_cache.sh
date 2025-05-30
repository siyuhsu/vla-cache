
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial    \
    --use_vla_cache True \