# ICP RL Walking Controller

Residual RL policy on top of an ICP (Instantaneous Capture Point) walking controller for a 3D bipedal walker in MuJoCo.

## Overview

The system uses a classical ICP-based walking controller as a base, and trains a residual RL policy (PPO) to improve tracking of commanded velocities. The RL policy outputs corrections added to the base controller's joint velocity commands.

## Setup

### 1. Create conda environment

```bash
conda create -n <name> python=3.11
conda activate <name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify MuJoCo

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

## Project Structure

```
├── walker_env.py          # Gymnasium environment (residual RL wrapper)
├── icp_controller.py      # ICP walking controller (base policy)
├── obs_manager.py         # Observation manager + command generator
├── train.py               # PPO training script (SB3)
├── utils.py               # Geometry / contact utilities
├── get_jacobian_3d_5dof_leg.py  # Analytical Jacobians
├── xml_files/
│   └── biped_3d_5dof_leg.xml   # MuJoCo robot model
├── requirements.txt
└── README.md
```

## Training

```bash
conda activate rl
python train.py
```

Training runs PPO with 10 parallel environments using `SubprocVecEnv`. 

### Configuration

Key settings in `train.py` you may want to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 10 | Parallel training environments |
| `steps_per_env` | 1,500,000 | Training steps per environment |
| `n_steps` | 2048 | Rollout length per env before PPO update |
| `batch_size` | 256 | Minibatch size for PPO |
| `eval_freq` | 10,000 | Evaluate every N steps |

### Monitoring

```bash
tensorboard --logdir ./tb_logs
```

Key metrics to watch:
- `rollout/ep_rew_mean` — average episode reward (main signal)
- `rollout/ep_len_mean` — average episode length (longer = staying alive)
- `train/entropy_loss` — should decrease slowly (not collapse to zero)

### Checkpoints

Checkpoints save to `./checkpoints/` every 100k steps. Best model (based on eval reward) saves to `./best_model/`.

