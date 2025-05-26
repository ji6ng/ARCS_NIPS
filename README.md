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
  pip install mujoco_py==0.5.7 gym==0.9.3 gym_compete==0.0.1 tensorflow==1.14 torch==1.10 numpy==1.19.5 stable-baseline==2.5.1
## Quick Start
1. Test our adversarial reward
 ```bash
 python -u ARCS/src/my_entrance.py --seed $SEED --mode 'llm' --env 'multicomp/SumoHumans-v0'
 ```
2. Launch the reward iteration script
 ```bash
   bash reward_iteration.sh
 ```
3. ### Running the Training Script

 **Edit `run.sh`**  
   - Set `mode` and `env_name` at the top of the script.  

     | Variable | Options | Purpose |
     |----------|---------|---------|
     | `mode` | `mask` \| `retrain` \| `oppo` \| `abs` \| `llm` | Selects the training behavior:<br>`mask` → train key-state identification module<br>`retrain` → fine-tune an existing policy<br>`oppo` → baseline 1<br>`abs` → baseline 2<br>`llm` → LLM-generated adversarial rewards |
     | `env_name` | any supported environment ID | Passed to `make_zoo_multi2single_env` |

 **Run the script**

   ```bash
   bash run.sh



