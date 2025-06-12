# Run VLA-Cache on OpenVLA

## Relevant Files

Evaluation
* `vla_cache_scripts`: VLA-Cache eval scripts
  * `run_openvla_with_vla_cache.sh`: VLA-Cache eval script
  * `run_openvla_without_vla_cache.sh`: Disable VLA-Cache eval utils
  * `download_model_openvla.sh`: Download checkpoints locally

Implementation

* `OpenVLA`: Inference core implementation
  * `src/openvla/prismatic/extern/hf/modeling_prismatic.py`: Modified the inference process

* [`transformers`](https://github.com/siyuhsu/transformers/tree/vla-cache-openvla): Dynamic cache update and LLAMA modelling implementation
  * `src/transformers/cache_utils.py`: Modified DynamicCache() class
  * `src/transformers/models/llama/modeling_llama.py`: Modified LlamaModel forward() function



## Setup

Set up a conda environment with LIBERO environment(follow instructions of OpenVLA in [README.md](README.md)).


Install OpenVLA dependencies from this project:

```bash
# activate conda environment
conda activate openvla

# install dependencies
cd src/openvla
pip install -e .
```

## VLA-Cache Evaluations Example

Download OpenVLA checkpoints for LIBERO-Spatial locally:

```bash
python vla_cache_scripts/download_model_local.py \
  --model_id openvla/openvla-7b-finetuned-libero-spatial
```


Run LIBERO-Spatial benchmark with VLA-Cache inference mode (Make sure the checkpoints in `src/openvla/checkpoints`. Don't load models from defalt huggingface cache path):

```bash
# Launch LIBERO-Spatial evals with VLA-Cache
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial  \
  --use_vla_cache True \
```

Run LIBERO-Spatial benchmark without VLA-Cache:

```bash
# Launch LIBERO-Spatial evals without VLA-Cache
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial  \
  --use_vla_cache False \
```
