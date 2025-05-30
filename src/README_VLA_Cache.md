# Run VLA-Cache on OpenVLA-OFT

## Relevant Files

Evaluation
* `vla_cache_scripts`: VLA-Cache eval scripts
  * `run_oft_with_vla_cache.sh`: VLA-Cache eval script
  * `run_oft_without_vla_cache.sh`: Disable VLA-Cache eval utils
  * `download_local_oft.sh`: Download checkpoints locally


## Setup

Set up a conda environment with LIBERO environment(follow instructions of OpenVLA-OFT in [SETUP.md](SETUP.md) and [LIBERO.md](LIBERO.md)).

Install transformers:

```bash
cd src/transformers
pip install -e .
```

Install OpenVLA-OFT:

```bash
cd src/openvla-oft
pip install -e .
```

Download OpenVLA-OFT checkpoints for LIBERO locally:

```bash
cd src/openvla-oft
bash vla_cache_scripts/download_local_oft.sh
```

## VLA-Cache Evaluations Example

Run LIBERO-Spatial benchmark with VLA-Cache inference mode (Make sure the checkpoints in `src/openvla-oft/checkpoints`):

```bash
# Launch LIBERO-Spatial evals with VLA-Cache
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial  \
  --use_vla_cache True \


# Launch LIBERO-Object evals with VLA-Cache
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object  \
  --use_vla_cache True \


# Launch LIBERO-Goal evals with VLA-Cache
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal  \
  --use_vla_cache True \


# Launch LIBERO-Long evals with VLA-Cache
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10  \
  --use_vla_cache True \
```

Run LIBERO-Spatial benchmark without VLA-Cache:

```bash
# Launch LIBERO-Spatial evals without VLA-Cache
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial  \
  --use_vla_cache False \
```
