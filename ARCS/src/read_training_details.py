import re
import ast
import pprint
from os.path import join

def parse_step_dicts(file_path):
    step_pattern = re.compile(r'\|\s*step\s*\|\s*(\d+)\s*\|', re.IGNORECASE)
    dict_pattern = re.compile(r'^\s*\{.*\}\s*$', re.S)

    result = {}
    current_step = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = step_pattern.search(line)
            if m:
                current_step = int(m.group(1))
                continue

            if dict_pattern.match(line):
                try:
                    metrics = ast.literal_eval(line.strip())
                except Exception:
                    continue
                if current_step is not None:
                    result[current_step] = metrics
                    current_step = None

    return result

if __name__ == '__main__':
    base_dir  = '/home/data/sdb5/jiangjunyong/ARCS/results/0521/'
    fname     = 'llm_sumo2.log'
    file_path = join(base_dir, fname)

    step_dicts = parse_step_dicts(file_path)

    # 直接去掉 sort_dicts 参数
    # pprint.pprint(step_dicts, width=120)
    print(step_dicts)
