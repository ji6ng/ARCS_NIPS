from LLM_query import query_llm
from typing import List
import os
import logging
import numpy as np
from RewardDic import Reward_Dic
class Reward_Evaluator(object):
    def __init__(self):
        self.GOAL_PROMPT = """The environment description is as follows: This environment simulates a competitive wrestling scenario in which two 3D bipedal robots engage in a match within a circular arena. Each robot consists of a torso (abdomen), a pair of arms, and a pair of legs, where each leg has three joints and each arm has two. The task is to control one of the robots by applying torques to its joints to defeat the opponent.
            The {observation space} consists of 395 dimensions and is structured as follows. The first 24 dimensions (\texttt{obs[0:24]}) represent the robot’s global position and the relative positions of its joints, including the torso’s global position (3D) and the rotational positions of the abdomen, hips, knees, shoulders, and elbows. The next 23 dimensions (\texttt{obs[24:47]}) store generalized velocity information, including the linear velocity of the torso and the angular velocities of all joints. Dimensions \texttt{obs[47:177]} describe the inertial properties of each major body part, including mass, center of mass position, and moments of inertia. Relative velocity information is recorded in \texttt{obs[177:255]}, where each body part has 6 dimensions representing linear and angular velocities. The actuator torques applied to each joint, which control the robot’s movement, are stored in \texttt{obs[255:278]}. External contact forces and torques applied to major body components are found in \texttt{obs[278:356]}. The opponent’s position state, structured identically to \texttt{obs[0:24]}, is stored in \texttt{obs[356:380]}. The next two dimensions (\texttt{obs[380:382]}) encode the relative distances between the two robots along the x and y axes. The torso’s rotation matrix, which defines its orientation in 3D space, is given in \texttt{obs[382:391]}. Finally, the last four dimensions (\texttt{obs[391:395]}) represent the radius of the arena, the robot’s distance to the boundary, the opponent’s distance to the boundary, and the remaining competition time.
            The \textbf{action space} consists of a 17-dimensional vector, where each element represents the torque applied to a joint, ranging from \texttt{-0.4} to \texttt{0.4} Nm. The action controls include rotational torques for the abdomen along three axes, rotations of the left and right hip joints, flexion of the left and right knee joints, as well as shoulder and elbow movements.
            The match outcome is determined by the following {victory conditions}. A robot wins if the opponent either falls (\texttt{z-coordinate of the torso < 1}) or exits the ring within the competition time. If neither condition is met before the time limit, the agent is considered to have lost.
            """
        self.TRAJECTORY = """The reward functions, reward component changes, and win rates are listed below.
        {details}"""
        self.PROMPT = """Based on the following reward functions, as well as the changes in reward components and win rates during training, identify the best-performing reward function. Only output the index number of the best reward function (e.g., 1, 2, 3...) based on improvements in reward components and win rates.
        {goal}
        {trajectory}"""
    def build_prompt(self, details: dict, failed : bool):
        prompt = self.PROMPT.format(
            goal=self.GOAL_PROMPT,
            trajectory=self.TRAJECTORY.format(details=details),
        )
        if failed:
            prefix = "Make sure to define a function def reward() in Python that exactly follows the arguments specified and returns one reward value.\n"
            prompt = prefix + prompt
        return prompt
    def evaluate_reward_func(self, details:dict):
        failed = False
        prompt = self.build_prompt(details, failed=failed)
        print("[P]:", prompt)
        code = query_llm(prompt)
        print(code)
        return code
