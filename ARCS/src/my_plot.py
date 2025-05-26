import re
import argparse
import matplotlib.pyplot as plt
from os.path import basename
import os
import numpy as np

def smooth(data, alpha=0.9):
    smoothed = data.copy()
    for i in range(len(smoothed) - 1):
        smoothed[i + 1] = smoothed[i] * alpha + smoothed[i + 1] * (1 - alpha) 
    return smoothed

def extract_metrics(file_path):
    pattern = re.compile(r'\|\s*([\w/]+)\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|')
    metrics = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            key = m.group(1).replace('/', '_')
            value = float(m.group(2))
            metrics.setdefault(key, []).append(value)
    return metrics

def plot_compare(all_metrics, labels, alpha=0.9, save_dir='pic/',len=None):
    common_keys = set(all_metrics[0].keys())
    for met in all_metrics[1:]:
        common_keys &= set(met.keys())

    for key in sorted(common_keys):
        plt.figure()
        for metrics, label in zip(all_metrics, labels):
            # if "reward" in key:
            #     print(key, metrics[key])
            data = metrics[key]
            sm = smooth(data, alpha)
            if len:
                sm = sm[:len]
            plt.plot(sm, label=label)
        plt.title(f'{key}')
        plt.xlabel('Record Index')
        plt.ylabel(key)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}{key}.png")
        plt.close()


def compare_paths(paths, alpha=0.9, save_dir='pic/', len=None):
    all_metrics = [ extract_metrics(p) for p in paths ]
    labels = [ os.path.basename(p) for p in paths ]
    plot_compare(all_metrics, labels, alpha=alpha, save_dir=save_dir,len=len)

def get_filepath(target_dir):
    file_list = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            file_list.append(os.path.join(root, f))
    print("递归获取的文件路径数组：")
    for file_path in file_list:
        print(file_path)
    return file_list

def compare_groups(paths, alpha=0.9, save_dir='pic/', length=None):
    # split into abs and llm groups
    groups = {'llm_sumo':[], 'sumo_abs':[],'sumo_oppo':[] } 
    for p in paths:
        name = os.path.basename(p)
        if 'llm_sumo' in name:
            groups['llm_sumo'].append(p)
        elif 'sumo_abs' in name:
            groups['sumo_abs'].append(p)
        elif 'sumo_oppo' in name:
            groups['sumo_oppo'].append(p)
    print(groups)

    # extract metrics per run
    group_metrics = {g: [extract_metrics(p) for p in ps] for g, ps in groups.items()}

    # find common metric keys across all runs
    common_keys = None
    for mets in group_metrics.values():
        for m in mets:
            if common_keys is None:
                common_keys = set(m.keys())
            else:
                common_keys &= set(m.keys())

    os.makedirs(save_dir, exist_ok=True)

    for key in sorted(common_keys):
        plt.figure()
        for g, mets in group_metrics.items():
            # collect smoothed series, truncated to 'length'
            series = []
            for m in mets:
                data = m[key]
                if length:
                    data = data[:length]
                series.append(smooth(data, alpha))
            # align lengths
            min_len = min(len(s) for s in series)
            arr = np.array([s[:min_len] for s in series])
            mean = arr.mean(axis=0)
            var = arr.var(axis=0)
            std = arr.std(axis=0)
            # sem = std / np.sqrt(arr.shape[0])
            xs = np.arange(min_len)
            plt.plot(xs, mean, label=f'{g} mean')
            plt.fill_between(xs, mean - std, mean + std, alpha=0.2)
        plt.title(key)
        plt.xlabel('Record Index')
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{key}_group.png"))
        plt.close()

if __name__ == '__main__':
    target_dir = "path/to/your/logs"  # replace with your actual path
    paths = get_filepath(target_dir)

    compare_paths(paths,
                  alpha=0.9,
                  save_dir='path/to/save/pic/',
                  len=3000)
    compare_groups( paths,
                    alpha=0.9,
                    save_dir='path/to/save/pic/',
                    length=3000
    )



