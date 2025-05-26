# ARCS_NIPS
## Installation  
1. **Install MuJoCo 1.3.1**  
   - Download MuJoCo 1.3.1 from [Roboti LLC](https://www.roboti.us/index.html)  
   - Unpack to `~/.mujoco/mujoco1.3.1/`  
   - Copy your `mjkey.txt` license file into `~/.mujoco/`  
   - Add to your `~/.bashrc` (or `~/.zshrc`):
     ```bash
     export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco1.3.1"
     export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco1.3.1/bin"
     ```

2. **Create a Python environment**  
   ```bash
   conda create -n rl-env python=3.7
   conda activate rl-env
  pip install mujoco_py==0.5.7
  pip install gym==0.9.3
  pip install gym_compete==0.0.1
  pip install tensorflow==1.14
  pip install torch==1.10
  pip install numpy==1.19.5

