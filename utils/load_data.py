
import json 
import pprint

from collections import defaultdict

from datasets import Dataset, load_dataset


def load_data_prm800k(data_dir, split='test'):
    data_by_levels = defaultdict(list)
    with open(f"{data_dir}/{split}.jsonl", 'r', encoding='utf-8') as fin:
        for line in fin:
            if line.strip():
                data = json.loads(line)
                # print(data['level'])
                data_by_levels[data['level']].append(data)
    
        # data =  [json.loads(line) for line in filein if line.strip()]
        # pprint.pprint(data, compact=True)
    
    for key in range(1,6):
        # key = str(key)
        print(f"{key}: {len(data_by_levels[key])}")
        # pprint.pprint(data_by_levels[key][:2], compact=True)
    # print(data_by_levels.keys())
    # pprint.pprint(data_by_levels['2'], compact=True)

    return data_by_levels


def load_data_prm800k_hf(data_dir, split='test'):
    dataset = load_dataset("json", data_files = f"{data_dir}/{split}.jsonl", split='train')
    return dataset