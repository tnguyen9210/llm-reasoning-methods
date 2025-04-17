
from utils import grader 
from datasets import load_dataset

def evaluate_correctness_trials(data_dir, data_by_levels, num_trials, num_budgets=None, start_idx=None):
        
    with open(data_dir, 'r', encoding='utf-8') as fin:
        # all_correctness = []
        all_correctness = np.zeros((num_trials, len(data_by_levels)))
        gt_answers = []
        trial_idx = 0
        for line in fin:
            if start_idx:
                if trial_idx < start_idx:
                    trial_idx += 1
                    continue
            if trial_idx >= num_trials:
                break
                
            trial_data = json.loads(line)
            for q_idx in range(len(data_by_levels)):
                if num_budgets is not None:
                    completions = trial_data['completions'][q_idx][:num_budgets]
                else:
                    completions = trial_data['completions'][q_idx]
            
                gt_answer = data_by_levels[q_idx]['answer']
                for cidx, completion in enumerate(completions):
                    c_answer = grader.extract_last_boxed_answer(completion)
                    is_correct = grader.grade_answer(c_answer, gt_answer):

                all_correctness[trial_idx][q_idx] = is_correct
                # all_correctness.append(is_correct)

            trial_idx += 1

    if start_idx:
        all_correctness = all_correctness[start_idx:]
        
    return all_correctness


def evaluate_correctness_hf(data_dir, dataset_name, config_name, dataset_split, level, num_budgets=None):

    dataset = load_dataset(dataset_name, name=config_name, split=dataset_split, cache_dir=data_dir)
    dataset_by_level = dataset.filter(lambda example: example['level'] == level)

    all_correctness = []
    for q_idx, data in enumerate(dataset_by_level):
        if num_budgets is not None:
            completions = data["completions"][:num_budgets]
        else:
            completions = data["completions"]

        gt_answer = data_by_levels[q_idx]['answer']
        
        for cidx, completion in enumerate(completions):
            c_answer = grader.extract_last_boxed_answer(completion)
            is_correct = grader.grade_answer(c_answer, gt_answer):
            if is_correct:
                break

        all_correctness.append(is_correct)

    return alll_correctness