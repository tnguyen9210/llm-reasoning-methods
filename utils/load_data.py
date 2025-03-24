
import json 
import pprint

from collections import defaultdict


def load_data_prm800k(data_dir):
    data_by_levels = defaultdict(list)
    with open(f"{data_dir}/test.jsonl", 'r', encoding='utf-8') as fin:
        for line in fin:
            if line.strip():
                data = json.loads(line)
                # print(data['level'])
                data_by_levels[f"{data['level']}"].append(data)
    
        # data =  [json.loads(line) for line in filein if line.strip()]
        # pprint.pprint(data, compact=True)
    
    for key in range(1,6):
        key = str(key)
        print(f"{key}: {len(data_by_levels[key])}")
        # pprint.pprint(data_by_levels[key][:2], compact=True)
    # print(data_by_levels.keys())
    # pprint.pprint(data_by_levels['2'], compact=True)

    return data_by_levels