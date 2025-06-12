# Run VLA-Cache on OpenVLA-OFT

## Relevant Files

Evaluation
* `vla_cache_scripts`: VLA-Cache eval scripts
  * `run_oft_with_vla_cache.sh`: VLA-Cache eval script
  * `run_oft_without_vla_cache.sh`: Disable VLA-Cache eval utils
  * `download_model_oft.sh`: Download checkpoints locally

Implementation

* `OpenVLA-OFT`: Inference core implementation
  * `src/openvla-oft/prismatic/extern/hf/modeling_prismatic.py`: Modified the inference process

* [`transformers`](https://github.com/siyuhsu/transformers/tree/vla-cache-openvla-oft): Dynamic cache update and LLAMA modelling implementation
  * `src/transformers/cache_utils.py`: Modified DynamicCache() class
  * `src/transformers/models/llama/modeling_llama.py`: Modified LlamaModel forward() function



## Setup

Set up a conda environment with LIBERO environment(follow instructions of OpenVLA-OFT in [SETUP.md](SETUP.md) and [LIBERO.md](LIBERO.md)).


Install OpenVLA-OFT dependencies from this project:

```bash
# activate conda environment
conda activate openvla-oft

# install dependencies
cd src/openvla-oft
pip install -e .
```

## VLA-Cache Evaluations Example

Download OpenVLA-OFT checkpoints for LIBERO-Spatial locally:

```bash
python vla_cache_scripts/download_model_local.py \
  --model_id moojink/openvla-7b-oft-finetuned-libero-spatial 
```

Run LIBERO-Spatial benchmark with VLA-Cache inference mode (Make sure the checkpoints in `src/openvla-oft/checkpoints`. Don't load models from defalt huggingface cache path):

```bash
# Launch LIBERO-Spatial evals with VLA-Cache
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial  \
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
