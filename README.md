# World Action Planner

This repository hosts the action-conditioned world models used in the **World Action Planner** paper.

> The world model architecture is based on [Large Video Planner](https://github.com/buoyancy99/large-video-planner) and the [Diffusion Forcing Transformer](https://github.com/kwsong0113/diffusion-forcing-transformer)

## Setup

### 1. Create and activate the mamba environment for simulators and planning

```bash
mamba create -n wap_env python=3.11 -y
mamba activate wap_env
```

### 2. Install all local environment packages

Install editable packages for all bundled environments:

```bash
pip install -e environments/robomimic
pip install -e environments/robosuite
pip install -e environments/LIBERO
```

### 3. Install the world-model client

```bash
pip install -e wm_client
```

### 4. Set up the world model

Follow the full setup guide in [`world_model/README.md`](world_model/README.md), including:
- editable install for `world_model` in a separate env
- checkpoint download
- Wan base file download
- server startup

### 5. Run the notebook for imagined actions

After setup is complete (and the world model server is running), open and run:

```bash
jupyter notebook demo.ipynb
```
