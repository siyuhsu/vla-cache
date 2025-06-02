# Run VLA-Cache on OpenVLA-OFT

## Relevant Files

Evaluation
* `vla_cache_scripts`: VLA-Cache eval scripts
  * `run_oft_with_vla_cache.sh`: VLA-Cache eval script
  * `run_oft_without_vla_cache.sh`: Disable VLA-Cache eval utils
  * `download_local_oft.sh`: Download checkpoints locally

Implementation

* `OpenVLA-OFT`: Inference core implementation
  * `src/openvla-oft/prismatic/extern/hf/modeling_prismatic.py`: Modified the inference process

* `transformers`: Dynamic cache update and LLAMA modelling implementation
  * `src/transformers/src/transformers/cache_utils.py`: Modified DynamicCache() class
  * `src/transformers/src/transformers/models/llama/modeling_llama.py`: Modified LlamaModel forward() function



## Setup

Set up a conda environment with LIBERO environment(follow instructions of OpenVLA-OFT in [SETUP.md](openvla-oft/SETUP.md) and [LIBERO.md](openvla-oft/LIBERO.md)).

1. Install transformers:

```bash
cd src/transformers
pip install -e .
```

2. Install OpenVLA-OFT:

```bash
cd src/openvla-oft
pip install -e .
```

3. Download OpenVLA-OFT checkpoints for LIBERO locally:

```bash
cd src/openvla-oft
bash vla_cache_scripts/download_local_oft.sh
```

NOTE: Make sure to follow the above steps to install openvla-oft and transformers, and download the hugging face model, otherwise the VLA-Cache implementation will not be effective.

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
