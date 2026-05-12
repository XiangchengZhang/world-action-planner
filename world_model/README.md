# World Models

Training, inference, and serving code for the action-conditioned world models used by World Action Planner.

## Quickstart (First-Time Setup)

Run all commands from this directory:

```bash
cd world_model
```

### 1. Create and activate a Python environment
We create a separate environment for the world model since we inference using websocket client
```bash
mamba create -n ei_world_model python=3.11 -y
mamba activate ei_world_model
```

### 2. Install project dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Install and authenticate Hugging Face CLI

```bash
pip install "huggingface_hub[cli]"
huggingface-cli login
```

### 4. Download released world-model checkpoints

```bash
mkdir -p data/ckpts
huggingface-cli download XiangchengZhang/world-action-planner \
  --include "world_models/**" \
  --local-dir data/ckpts \
  --local-dir-use-symlinks False

mv data/ckpts/world_models/* data/ckpts/
rmdir data/ckpts/world_models
```

Expected default checkpoint:

```text
data/ckpts/libero_90_base/checkpoints/latest.ckpt
```

Also included in the release:

```text
data/ckpts/libero_object_ft/checkpoints/latest.ckpt
data/ckpts/robosuite_ft/checkpoints/latest.ckpt
data/ckpts/robosuite_default_prompt.pt
data/ckpts/robosuite_default_neg_prompt.pt
```

### 5. Download required Wan base files

The released `libero_90_base` checkpoint uses precomputed prompt embeddings. You only need the Wan config and VAE file:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --include "config.json" "Wan2.1_VAE.pth" \
  --local-dir data/ckpts/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False
```

Expected files:

```text
data/ckpts/Wan2.1-T2V-1.3B/config.json
data/ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

If you run image-to-video configs, also download:

```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir data/ckpts/Wan2.1-I2V-14B-480P \
  --local-dir-use-symlinks False
```
For training from the original Wan T2V checkpoint, you should also download the Wan-T2V-1.3B checkpoint
## Run the World Model Server

Start with the default released checkpoint:

```bash
python server.py --host 0.0.0.0 --port 7880
```

Use a specific checkpoint path:

```bash
python server.py \
  --checkpoint-path data/ckpts/robosuite_ft/checkpoints/latest.ckpt \
  --host 0.0.0.0 \
  --port 7880
```

Or set the checkpoint via environment variable:

```bash
export WORLD_MODEL_PATH=data/ckpts/libero_object_ft/checkpoints/latest.ckpt
python server.py
```

WebSocket request format:

```json
{
  "history_frames": ["<base64 png>", "..."],
  "history_conds": ["<base64 png>", "..."],
  "future_conds": ["<base64 png>", "..."]
}
```

Response fields: `full_video`, `pred_frames`, and `pred_panels` (all base64 PNG frames).

## Configuration Notes

Main Hydra entrypoint: [main.py](main.py)  
Defaults: [configurations/config.yaml](configurations/config.yaml)

Current local default config:

```text
experiment=exp_video
dataset=robosuite
algorithm=wan_at2v
cluster=null
```

For serving, [pipeline.py](pipeline.py) loads `.hydra/config.yaml` next to the checkpoint. Keep released checkpoint folders intact:

```text
data/ckpts/libero_90_base/.hydra/config.yaml
data/ckpts/libero_90_base/checkpoints/latest.ckpt
```

## Troubleshooting

- If model download fails, re-run `huggingface-cli login` and verify access to `XiangchengZhang/world-action-planner`.
- If CUDA is unavailable, import checks can still pass, but full Wan inference requires a GPU.
- If Matplotlib cache permissions fail on shared systems:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-$USER
```
