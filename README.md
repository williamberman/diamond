# Diffusion for World Modeling: Visual Details Matter in Atari

**TL;DR** We introduce DIAMOND (DIffusion As a Model Of eNvironment Dreams), a reinforcement learning agent trained in a diffusion world model.

<div align='center'>
  Autoregressive imagination with DIAMOND on a subset of Atari games
<img alt="DIAMOND agent in WM" src="assets/main.gif">
</div>

Quick install to try our [pretrained world models](#try) using [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/):

>```bash
>git clone git@github.com:eloialonso/diamond.git
>cd diamond
>conda create -n diamond python=3.10
>conda activate diamond
>pip install -r requirements.txt
>python src/play.py --pretrained
>```

Alternatively, if you do not have miniconda installed you can use python [venv](https://docs.python.org/3/library/venv.html):
>```bash
>git clone git@github.com:eloialonso/diamond.git
>cd diamond
>python3 -m venv diamond_env
>source activate ./diamond_env/bin/
>pip install -r requirements.txt
>python src/play.py --pretrained
>```
And press `m` to take control (the policy is playing by default)!

**Warning**: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

<a name="quick_links"></a>
## Quick Links

- [Try our playable diffusion world models](#try)
- [Launch a training run](#launch)
- [Configuration](#configuration)
- [Visualization](#visualization)
  - [Play mode (default)](#play_mode)
  - [Dataset mode (add `-d`)](#dataset_mode)
  - [Other options, common to play/dataset modes](#other_options)
- [Run folder structure](#structure)
- [Results](#results)
- [Citation](#citation)
- [Credits](#credits)

<a name="try"></a>
## [⬆️](#quick_links) Try our playable diffusion world models

```bash
python src/play.py --pretrained
```

Then select a game, and world model and policy pretrained on Atari 100k will be downloaded from our [repository on Hugging Face Hub 🤗](https://huggingface.co/eloialonso/diamond) and cached on your machine.

Some things you might want to try:
- Press `m` to change the policy between the agent and human (the policy is playing by default).
- Press `↑/↓` to change the imagination horizon (default is 50 for playing).

To adjust the sampling parameters (number of denoising steps, stochasticity, order, etc) of the trained diffusion world model, for instance to trade off sampling speed and quality, edit the section `world_model_env.diffusion_sampler` in the file `config/trainer.yaml`.

See [Visualization](#visualization) for more details about the available commands and options.

<a name="launch"></a>
## [⬆️](#quick_links) Launch a training run

To train with the hyperparameters used in the paper, launch:
```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.device=cuda:0
```

This creates a new folder for your run, located in `outputs/YYYY-MM-DD/hh-mm-ss/`.

To resume a run that crashed, navigate to the fun folder and launch:

```bash
./scripts/resume.sh
```

<a name="configuration"></a>
## [⬆️](#quick_links) Configuration

We use [Hydra](https://github.com/facebookresearch/hydra) for configuration management.

All configuration files are located in the `config` folder:

- `config/trainer.yaml`: main configuration file.
- `config/agent/default.yaml`: architecture hyperparameters.
- `config/env/atari.yaml`: environment hyperparameters.

You can turn on logging to [weights & biases](https://wandb.ai) in the `wandb` section of `config/trainer.yaml`.

Set `training.model_free=true` in the file `config/trainer.yaml` to "unplug" the world model and perform standard model-free reinforcement learning.

<a name="visualization"></a>
## [⬆️](#quick_links) Visualization

<a name="play_mode"></a>
### [⬆️](#quick_links) Play mode (default)

To visualize your last checkpoint, launch **from the run folder**:

```bash
python src/play.py
```

By default, you visualize the policy playing in the world model. To play yourself, or switch to the real environment, use the controls described below.

```txt
Controls (play mode)

(Game-specific commands will be printed on start up)

⏎   : reset environment

m   : switch controller (policy/human)
↑/↓ : imagination horizon (+1/-1)
←/→ : next environment [world model ←→ real env (test) ←→ real env (train)]

.   : pause/unpause
e   : step-by-step (when paused)
```

Add `-r` to toggle "recording mode" (works only in play mode). Every completed episode will be saved in `dataset/rec_<env_name>_<controller>`. For instance:

- `dataset/rec_wm_π`: Policy playing in world model.
- `dataset/rec_wm_H`: Human playing in world model.
- `dataset/rec_test_H`: Human playing in test real environment.

You can then use the "dataset mode" described in the next section to replay the stored episodes.

<a name="dataset_mode"></a>
### [⬆️](#quick_links) Dataset mode (add `-d`)

**In the run folder**, to visualize the datasets contained in the `dataset` subfolder, add `-d` to switch to "dataset mode":

```bash
python src/play.py -d
```

You can use the controls described below to navigate the datasets and episodes.

```txt
Controls (dataset mode)

m   : next dataset (if multiple datasets, like recordings, etc)
↑/↓ : next/previous episode
←/→ : next/previous timestep in episodes
PgUp: +10 timesteps
PgDn: -10 timesteps
⏎   : back to first timestep
```

<a name="other_options"></a>
### [⬆️](#quick_links) Other options, common to play/dataset modes

```txt
--fps FPS             Target frame rate (default 15).
--size SIZE           Window size (default 800).
--no-header           Remove header.
```

<a name="structure"></a>
## [⬆️](#quick_links) Run folder structure

Each new run is located at `outputs/YYYY-MM-DD/hh-mm-ss/`. This folder is structured as follows:

```txt
outputs/YYYY-MM-DD/hh-mm-ss/
│
└─── checkpoints
│   │   state.pt  # full training state
│   │
│   └─── agent_versions
│       │   ...
│       │   agent_epoch_00999.pt
│       │   agent_epoch_01000.pt  # agent weights only
│
└─── config
│   |   trainer.yaml
|
└─── dataset
│   │
│   └─── train
│   |   │   info.pt
│   |   │   ...
|   |
│   └─── test
│       │   info.pt
│       │   ...
│
└─── scripts
│   │   resume.sh
|   |   ...
|
└─── src
|   |   main.py
|   |   ...
|
└─── wandb
    |   ...
```

<a name="results"></a>
## [⬆️](#quick_links) Results

The file [results/data/DIAMOND.json](results/data/DIAMOND.json) contains the results for each game and seed used in the paper.

The DDPM code used for Section 5.1 of the paper can be found on the [ddpm](https://github.com/eloialonso/diamond/tree/ddpm) branch.

<a name="citation"></a>
## [⬆️](#quick-links) Citation

```text
@misc{alonso2024diffusion,
      title={Diffusion for World Modeling: Visual Details Matter in Atari},
      author={Eloi Alonso and Adam Jelley and Vincent Micheli and Anssi Kanervisto and Amos Storkey and Tim Pearce and François Fleuret},
      year={2024},
      eprint={2405.12399},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<a name="credits"></a>
## [⬆️](#quick_links) Credits

- [https://github.com/crowsonkb/k-diffusion/](https://github.com/crowsonkb/k-diffusion/)
- [https://github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
