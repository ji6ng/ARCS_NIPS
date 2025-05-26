import os
import pickle
from reward_evaluator import Reward_Evaluator

def load_training_details(folder_path):
    all_data = {}
    for root, _, files in os.walk(folder_path):
        for fname in files:
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, folder_path)
            with open(path, 'rb') as f:
                details = pickle.load(f)
                reward_str = pickle.load(f)
                all_data[rel] = {
                        'details': details,
                        'reward_str': reward_str
                    }
    return all_data

if __name__ == "__main__":
    folder = "/home/data/sdb5/jiangjunyong/ARCS/results/reward"
    all_details = load_training_details(folder)
    RewardEvaluator = Reward_Evaluator()
    x = RewardEvaluator.evaluate_reward_func(all_details)
    best = all_details['training_%d.pkl'%int(x)]
    with open(f"/home/data/sdb5/jiangjunyong/ARCS/results/best.pkl", "wb") as f:
        pickle.dump(best["details"], f)
        pickle.dump(best["reward_str"], f)