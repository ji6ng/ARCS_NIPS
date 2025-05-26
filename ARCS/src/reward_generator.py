from LLM_query import query_llm
from typing import List
import os
import logging
import numpy as np
from RewardDic import Reward_Dic
import torch

class Reward_Generator(object):
    """
    Class for generating shaped reward function.
    """

    def __init__(self, save_dir:str):
        self.GOAL_PROMPT = """
The environment simulates a wrestling scenario between two 3D bipedal robots in a circular arena. Each robot consists of a torso (abdomen), a pair of arms, and a pair of legs. Each leg has three joints, and each arm has two joints. The objective is to control the joint torques of one robot to defeat the opponent.
The **observation space** (395 dimensions) includes the following components:
* **obs\[0:24]**: Represents the robot's global position and relative joint positions. This includes the global coordinates of the torso (3 dimensions) and rotational positions of each joint:
  * Abdomen (`abdomen_x`, `abdomen_y`, `abdomen_z`, 1 dimension each),
  * Right hip (`right_hip_x`, `right_hip_y`, `right_hip_z`, 1 dimension each),
  * Right knee (1 dimension),
  * Left hip (`left_hip_x`, `left_hip_y`, `left_hip_z`, 1 dimension each),
  * Left knee (1 dimension),
  * Right shoulder (`right_shoulder1`, `right_shoulder2`, 1 dimension each),
  * Right elbow (1 dimension),
  * Left shoulder (`left_shoulder1`, `left_shoulder2`, 1 dimension each),
  * Left elbow (1 dimension).
* **obs\[24:47]**: Generalized velocity information, including the torso’s linear velocity (3 dimensions) and the angular velocities of each joint.
* **obs\[47:177]**: Describes the inertial properties of each major body part, including mass, center of mass, and inertia tensor. Each part occupies 10 dimensions.
* **obs\[177:255]**: Relative velocity information of major body parts, with each part represented by 6 dimensions (linear and angular velocity).
* **obs\[255:278]**: Actuator torques applied to each joint, used to control joint rotation or swinging.
* **obs\[278:356]**: External contact forces and torques on major body parts, with 6 dimensions per part (3 linear forces and 3 torques).
* **obs\[356:380]**: Opponent robot's position state, with the same structure as obs\[0:24].
* **obs\[380:382]**: Relative position between the two robots along the `x` and `y` axes.
* **obs\[382:391]**: Rotation matrix of the torso, representing its orientation in 3D space.
* **obs\[391:395]**: The last four dimensions represent the arena radius, the distance from our robot to the arena boundary, the distance from the opponent to the boundary, and the remaining match time.
The **action space** is a 17-dimensional vector, where each element corresponds to the torque applied to a specific joint. The torque values range from -0.4 to 0.4 Nm. The actions include rotational torques along three axes of the abdomen, rotations of the left and right hip joints, bending of the left and right knees, two-directional rotations of the right shoulder, bending of the right elbow, two-directional rotations of the left shoulder, and bending of the left elbow — enabling control over the robot’s posture and gait.
**Victory condition**: A robot wins if the opponent falls (i.e., its torso z-coordinate is less than 1) or steps out of the arena. If no winner is determined within the time limit, the opponent is considered the winner.
        """
        self.TRAJECTORY = """
Your previous reward function was:
{code}
During 30 rounds of PPO training, we evaluated the performance before training and after every 6 rounds. The resulting reward component dictionary recorded at each checkpoint is as follows:
{details},
Note: Each value in the dictionary represents the accumulated reward per episode, not per step.
Among these components:
step indicates the average number of steps per episode.
winning_rate represents the agent's success rate, which is the most critical metric to optimize.
Please analyze the feedback in the reward dictionary carefully and design a new, improved reward function to better address the task and increase the success rate. Below are some useful guidelines for interpreting the current reward components:
Necessity of Redesign:
If the success rate (winning_rate) remains close to zero, the entire reward function needs to be redesigned to explicitly encourage goal-oriented behaviors.
Improving Non-Optimizable Reward Components:
If a reward component’s value remains nearly constant, reinforcement learning may struggle to optimize it. Consider the following:
(a) Rescale it or apply a temperature parameter;
(b) Redesign the component to make it more discriminative;
(c) Remove components that do not contribute to performance.
Balancing Reward Magnitudes:
If certain reward components are significantly larger than others, you should scale them to avoid dominance over the total reward.
Importantly, you must ensure that the reward encouraging the agent to run toward the goal slightly dominates other components.
Apply nonlinear transformations (e.g., torch.exp, normalization) to smooth reward values, and introduce temperature parameters to control their effect scale.
Reward Function Output Structure:
Your reward function should return two parts:
Total reward: a scalar used as the optimization target in reinforcement learning.
Reward component dictionary: clearly indicating the value of each sub-reward, for future analysis and refinement.
Strategies for Designing an Improved Reward Function:
Utilize all input parameters (s1, s2, victory_status) to ensure sensitivity to state transitions.
Encourage the agent to win through proactive and exploratory behaviors.
Introduce appropriate penalties to discourage passive or wasteful actions.
Give distinct reward differences between victory and defeat.
Key Design Considerations:
Time constraints: Ensure that the agent aims to finish the match successfully within the allowed duration.
Reward balance: Adjust the weight of each component so that no single one overwhelms the total reward.
Output format: Return both the total reward and the detailed reward component dictionary for iterative tuning and analysis.
        """
        self.PROMPT = """
When I ask you a question, only respond with the code that answers it—do not include any additional text before or after the code. Your objective is to improve the reinforcement learning model’s success rate (winning_rate) and ensure that the agent actively defeats its opponent within the competition time limit.
To achieve this goal, design a reward function that helps the agent learn a winning strategy more quickly and stably. Specifically, write a Python class named RewardCalculator, where the compute() method serves as the reward function. This function should calculate the reward based on the following parameters:
s1: the agent’s previous state,
s2: the agent’s current state,
a2: the current action taken,
victory_status: a string indicating the victory result, with possible values "win", "loss", or None,
s_o: the opponent’s state after being affected by a2,
reward_adv: the agent’s base reward (between -10 and 10),
reward_opp: the opponent’s base reward (between -10 and 10).
You must design a reward based on the transition from s1 to s2 and the action a2.
Do not output anything other than the RewardCalculator class.
Ensure that the compute() function returns both the total reward and a dictionary of component rewards (reflecting the metrics you considered).
The reward function must depend on all the input parameters and exhibit a certain degree of complexity.
The core objective of this reward function is to increase the agent’s success rate (winning_rate) and ensure that it achieves victory within the competition time limit.

{goal}
{trajectory}
        """
        self.PROMPT_default = """
When I ask you a question, only respond with the code that answers it—do not include any additional text before or after the code. Your objective is to improve the reinforcement learning model’s success rate (winning_rate) and ensure that the agent actively defeats its opponent within the competition time limit.
To achieve this goal, design a reward function that helps the agent learn a winning strategy more quickly and stably. Specifically, write a Python class named RewardCalculator, where the compute() method serves as the reward function. This function should calculate the reward based on the following parameters:
s1: the agent’s previous state,
s2: the agent’s current state,
a2: the current action taken,
victory_status: a string indicating the victory result, with possible values "win", "loss", or None,
s_o: the opponent’s state after being affected by a2,
reward_adv: the agent’s base reward (between -10 and 10),
reward_opp: the opponent’s base reward (between -10 and 10).
You must design a reward based on the transition from s1 to s2 and the action a2.
Do not output anything other than the RewardCalculator class.
Ensure that the compute() function returns both the total reward and a dictionary of component rewards (reflecting the metrics you considered).
The reward function must depend on all the input parameters and exhibit a certain degree of complexity.
The core objective of this reward function is to increase the agent’s success rate (winning_rate) and ensure that it achieves victory within the competition time limit.

{goal}
        """
        self.log_of_responses = []
        self.valid_code_history = ['none']
        self.save_dir = save_dir
    
    def build_prompt(self, code, details: dict, failed : bool):
        prompt = self.PROMPT.format(
            goal=self.GOAL_PROMPT,
            # param=self.PARAM_PROMPT,
            trajectory=self.TRAJECTORY.format(code=code, details=details),
        )

        if failed:
            prefix = "Make sure to define a function def reward() in Python that exactly follows the arguments specified and returns one reward value.\n"
            prompt = prefix + prompt
        return prompt

    def generate_default_func(self):
        """
        Default reward function.
        """
        code = Reward_Dic['multicomp/SumoHumans-v0']
        self.valid_code_history.append(code)
        return code

    def generate_reward_func(self, codes, details:dict):
        """
        Current format hard-coded for MountainCar or Showdown to work and produce less errors.
        """
        # while True:
        failed = False
        # trajectory = trajectory[-50:]
        code = None
        for i in range(2):
            try:
                prompt = self.build_prompt(codes, details, failed=failed)
                print("[P]:", prompt)
                code = query_llm(prompt)
                self.log_of_responses.append({"prompt": prompt, "code": code})
                logging.info('code', code)
                print(code)
                local_scope = {}
                exec(code, globals(), local_scope)
            except:
                self.valid_code_history.append('failed: ' + code)
                if failed:
                    print("Error: failed again!")
                    print(code)
                else:
                    print("Error in trying to define function!")
                failed = True
                continue
            self.valid_code_history.append(code)
            return code
        return code

    def dump(self):
        print(self.log_of_responses)

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'reward_code.txt'), 'w') as f:
            f.write(str(self.valid_code_history))

    def build_prompt_default(self):
        prompt = self.PROMPT_default.format(
            goal=self.GOAL_PROMPT,
        )
        return prompt
    def generate_reward_func_default(self):
        """
        Current format hard-coded for MountainCar or Showdown to work and produce less errors.
        """
        code = None
        for i in range(2):
            try:
                prompt = self.build_prompt_default()
                print("[P]:", prompt)
                code = query_llm(prompt)
                self.log_of_responses.append({"prompt": prompt, "code": code})
                logging.info('code', code)
                print(code)
                local_scope = {}
                exec(code, globals(), local_scope)
            except:
                self.valid_code_history.append('failed: ' + code)
                if failed:
                    print("Error: failed again!")
                    print(code)
                else:
                    print("Error in trying to define function!")
                failed = True
                continue
            self.valid_code_history.append(code)
            return code
        return code

