import signal

import random
import numpy as np
np.set_printoptions(precision=4)
from scipy.stats import ttest_rel

from datasets import load_dataset
from utils import parser, grader2

import logging


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def run_with_timeout(fn_extract_answer, fn_grade, completion, gt_answer, timeout=2):
    # Set the signal handler for SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Schedule an alarm after `timeout` seconds
    try:
        c_answer = fn_extract_answer(completion, 'math')
        # print(c_answer)
        result = fn_grade(c_answer, gt_answer)
    except TimeoutException:
        print(f"Timeout: {completion}")
        c_answer = None
        result = None
    finally:
        signal.alarm(0)  # Cancel alarm if function returns early
    return c_answer, result



def compute_correctness_curve_ncomps_hf(data_dir, level, max_ncomps, timeout=2):

    # dataset = load_dataset(dataset_name, name=config_name, split=dataset_split, cache_dir=data_dir)
    dataset = load_dataset("json", data_files = data_dir, split='train')
    dataset_by_level = dataset.filter(lambda example: example['level'] == level)

    # N = 100 # number of completions 
    peak1b_correctness = np.zeros((len(dataset_by_level), max_ncomps))
    peak1b_idxes = np.zeros((len(dataset_by_level), max_ncomps))
    for q_idx, data in enumerate(dataset_by_level):
        # print(f"q_idx = {q_idx}")
        pass1b_completions = data["completions"]

        # gt_answer = data['answer']
        gt_cot, gt_answer  = parser.parse_ground_truth(data, 'math')

        # print(f"done3")
        max_idxes_list = []
        max_correctness_list = []
        max_score = float('-inf')
        max_idx = -1
        max_completion = None
        agg_scores = data["agg_scores"]
        if len(pass1b_completions) == 0:
            continue
            
        for cidx, (completion, score) in enumerate(zip(pass1b_completions, agg_scores)):
            if score > max_score:
                max_score = score
                max_idx = cidx
                max_completion = completion
                
                max_c_answer, max_is_correct = \
                    run_with_timeout(parser.extract_answer, grader2.math_equal, max_completion, gt_answer)
                
            max_idxes_list.append(max_idx)
            max_correctness_list.append(max_is_correct)

        # print(len(pass1b_completions))
        # print(peak1b_best_idx)
        # print(peak1b_is_correct)
        if len(max_correctness_list) < max_ncomps:
            # print(q_idx, len(peak1b_is_correct), len(pass1b_completions))
            max_idxes_list_ext = max_idxes_list + [max_idxes_list[-1]]*(max_ncomps - len(max_idxes_list))
            max_correctness_list_ext = max_correctness_list + [max_correctness_list[-1]]*(max_ncomps - len(max_correctness_list))

        else:
            logging.error(len(max_correctness_list))

        peak1b_idxes[q_idx,:] = max_idxes_list_ext
        peak1b_correctness[q_idx,:] = max_correctness_list_ext
        
        
    return peak1b_correctness, peak1b_idxes

def compute_correctness_curve_budget(dataset, step_budget, timeout=2):

    # dataset = load_dataset(dataset_name, name=config_name, split=dataset_split, cache_dir=data_dir)
    # dataset = load_dataset("json", data_files = data_dir, split='train')
    # dataset_by_level = dataset.filter(lambda example: example['level'] == level)
    # selected_cidxes = [10, 12, 46, 49, 79, 82, 96]

    # N = 100 # number of completions 
    peak1b_correctness = np.zeros((len(dataset), step_budget))
    peak1b_idxes = np.zeros((len(dataset), step_budget))
    for q_idx, data in enumerate(dataset):
        # if q_idx not in selected_cidxes:
        #     continue
        # logging.error(f"\n-> q_idx = {q_idx}")
        pass1b_completions = data["completions"]
        pass1b_csteps = data["csteps"]

        # gt_answer = data['answer']
        gt_cot, gt_answer  = parser.parse_ground_truth(data, 'math')

        # print(f"done3")
        max_idxes_list = []
        max_correctness_list = []
        max_score = float('-inf')
        max_idx = -1
        max_completion = None
        max_step_cnt = -1
        max_overlap = False
        agg_scores = data["agg_scores"]
        if len(pass1b_completions) == 0:
            continue
            
        for cidx, (completion, step_cnt, score) in enumerate(zip(pass1b_completions, pass1b_csteps, agg_scores)):
            # logging.error(f"\n-> cidx = {cidx}")
            # logging.error(f"score = {score:0.4f}")
            # logging.error(f"max_score = {max_score:0.4f}")
            # logging.error(f"step_cnt = {step_cnt}")
            # tmp = completion.split("\n\n")
            # logging.error(f"comp = {tmp[-1]}")
            if score > max_score:
                max_score = score
                max_idx = cidx
                max_completion = completion
                if step_cnt == max_step_cnt:
                    logging.fatal(f"q_idx = {q_idx}")
                    max_overlap = True
                max_step_cnt = step_cnt
                
                max_c_answer, max_is_correct = \
                    run_with_timeout(parser.extract_answer, grader2.math_equal, max_completion, gt_answer)

            # logging.error(f"max_score = {max_score:0.4f}")
            # logging.error(f"max_idx = {max_idx}")
            # max_idxes_list.append(max_idx)
            # max_correctness_list.append(max_is_correct)
            if max_overlap:
                max_correctness_list[-1] = max_is_correct
            else: 
                max_correctness_list += [max_is_correct]*(step_cnt - len(max_correctness_list))

            max_overlap = False 
            # logging.error(f"max_is_correct = {max_is_correct}")
            # logging.error(max_correctness_list)
            # logging.error(len(max_correctness_list))

        # print(len(pass1b_completions))
        # print(max_correctness_list)
        
        if len(max_correctness_list) < step_budget:
            # print(q_idx, len(peak1b_is_correct), len(pass1b_completions))
            # max_idxes_list_ext = max_idxes_list + [max_idxes_list[-1]]*(max_ncomps - len(max_idxes_list))
            max_correctness_list += [max_correctness_list[-1]]*(step_budget - len(max_correctness_list))

        else:
            logging.error(len(max_correctness_list))

        # peak1b_idxes[q_idx,:] = max_idxes_list_ext
        peak1b_correctness[q_idx,:] = max_correctness_list
        logging.error(f"final: {max_correctness_list}")

    return peak1b_correctness, peak1b_idxes

def max_with_index(arr):
    max_score = arr[0]
    max_idx = 0
    for i, val in enumerate(arr):
        if val > max_score:
            max_score = val
            max_idx = i
    return max_score, max_idx


def compute_stats_correctness_curve_budget(result_dir, config_name, num_trials, step_budget):
    
    all_peak1b_correctness = []
    
    for trial_idx in range(num_trials):
        # load data
        dataset_res = load_dataset("json", data_files = f"{result_dir}/{config_name}--trial-{trial_idx}.jsonl", split='train')
        
        peak1b_correctness, peak1b_idxes = compute_correctness_curve_budget(dataset_res, step_budget)

        all_peak1b_correctness.append(peak1b_correctness)

    all_peak1b_correctness = np.concatenate(all_peak1b_correctness)
    print(all_peak1b_correctness.shape)
    # print(all_peak1b_correctness)

    nsamples = len(all_peak1b_correctness)
    peak1b_correctnes_mean = np.mean(all_peak1b_correctness, axis=0)
    peak1b_correctness_std = np.std(all_peak1b_correctness, axis=0, ddof=1)/np.sqrt(nsamples)

    # print(all_peak1b_avg_mean)
    peak1b_max_mean, peak1b_max_idx = max_with_index(peak1b_correctnes_mean) 
    print(f"peak1b_score = {peak1b_max_mean:0.4f} (\u00B1{peak1b_correctness_std[peak1b_max_idx]:0.4f})")

    return peak1b_correctnes_mean, peak1b_correctness_std

def evaluate_correctness(dataset, timeout=2):
    
    passn_correctness = np.zeros((len(dataset)))
    pass1b_correctness = np.zeros((len(dataset)))
    naive1b_correctness = np.zeros((len(dataset)))
    weighted1b_correctness = np.zeros((len(dataset)))
    maj1b_correctness = np.zeros((len(dataset)))
    
    pass1b_ncomps = np.zeros((len(dataset)))
    pass1b_lengths = np.zeros((len(dataset)))

    pass1b_nphases = np.zeros((len(dataset)))
    pass1b_ndepths = np.zeros((len(dataset)))

    for q_idx, data in enumerate(dataset):
        # print(f"q_idx = {q_idx}")
        # passn_completions = data["completions"][:n]
        

        gt_cot, gt_answer  = parser.parse_ground_truth(data, 'math')
        naive1b_answer = parser.extract_answer(data[f"pred_naive@{0}"], 'math')
        weighted1b_answer = parser.extract_answer(data[f"pred_weighted@{0}"], 'math')
        maj1b_answer = parser.extract_answer(data[f"pred_maj@{0}"], 'math')

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # Schedule an alarm after `timeout` seconds
        try:
            naive1b_correct = grader2.math_equal(naive1b_answer, gt_answer)
            # print(c_answer)
        except TimeoutException:
            print(f"Timeout: {completion}")
            naive1b_correct = False
        finally:
            signal.alarm(0)  # Cancel alarm if function returns early

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # Schedule an alarm after `timeout` seconds
        try:
            weighted1b_correct = grader2.math_equal(weighted1b_answer, gt_answer)
            # print(c_answer)
        except TimeoutException:
            weighted1b_correct = False
        finally:
            signal.alarm(0)  # Cancel alarm if function returns early

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # Schedule an alarm after `timeout` seconds
        try:
            maj1b_correct = grader2.math_equal(maj1b_answer, gt_answer)
            # print(c_answer)
        except TimeoutException:
            maj1b_correct = False
        finally:
            signal.alarm(0)  # Cancel alarm if function returns early

        pass1b_completions = data["completions"]

        passn_correct = False
        pass1b_correct = False
        for cidx, completion in enumerate(pass1b_completions):
            c_answer, is_correct = run_with_timeout(parser.extract_answer, grader2.math_equal, completion, gt_answer)
            if is_correct is True: 
                passn_correct = True
                pass1b_correct = True
                break

        passn_correctness[q_idx] = passn_correct
        pass1b_correctness[q_idx] = pass1b_correct
        naive1b_correctness[q_idx] = naive1b_correct
        weighted1b_correctness[q_idx] = weighted1b_correct
        maj1b_correctness[q_idx] = maj1b_correct

        pass1b_ncomps[q_idx] = len(pass1b_completions)
        pass1b_lengths[q_idx] = np.mean(data["cdepths"]) 
        
        # print(data["cdepths"])
        # print(pass1b_lengths[q_idx])

        pass1b_nphases[q_idx] = data["last_phase"]
        pass1b_ndepths[q_idx] = np.mean(data["tdepths"])
            
    return passn_correctness, pass1b_correctness, naive1b_correctness, weighted1b_correctness, maj1b_correctness, pass1b_ncomps, pass1b_lengths, \
        pass1b_nphases, pass1b_ndepths

def compute_stats_completions(result_dir, config_name, num_trials):
    all_passn_correctness = []
    all_pass1b_correctness = []
    all_naive1b_correctness = []
    all_weighted1b_correctness = []
    all_maj1b_correctness = []
    
    all_pass1b_ncomps = []
    all_pass1b_lengths = []
    all_pass1b_nphases = []
    all_pass1b_ndepths = []
    
    for trial_idx in range(num_trials):
        # load data
        dataset_res = load_dataset("json", data_files = f"{result_dir}/{config_name}--trial-{trial_idx:03d}.jsonl", split='train')
        
        passn_correctness, pass1b_correctness, naive1b_correctness, weighted1b_correctness, maj1b_correctness, \
            pass1b_ncomps, pass1b_lengths, pass1b_nphases, pass1b_ndepths = evaluate_correctness(dataset_res)
    
        # all_passn_correctness.append(passn_correctness)
        all_pass1b_correctness.append(pass1b_correctness)
        all_naive1b_correctness.append(naive1b_correctness)
        all_weighted1b_correctness.append(weighted1b_correctness)
        all_maj1b_correctness.append(maj1b_correctness)
    
        all_pass1b_ncomps.append(pass1b_ncomps)
        all_pass1b_lengths.append(pass1b_lengths)
        all_pass1b_nphases.append(pass1b_ncomps)
        all_pass1b_ndepths.append(pass1b_lengths)

    # all_passn_correctness = np.concatenate(all_passn_correctness)
    all_pass1b_correctness = np.concatenate(all_pass1b_correctness)
    all_naive1b_correctness = np.concatenate(all_naive1b_correctness)
    all_weighted1b_correctness = np.concatenate(all_weighted1b_correctness)
    all_maj1b_correctness = np.concatenate(all_maj1b_correctness)
    
    all_pass1b_ncomps = np.concatenate(all_pass1b_ncomps)
    all_pass1b_lengths = np.concatenate(all_pass1b_lengths)
    all_pass1b_nphases = np.concatenate(all_pass1b_nphases)
    all_pass1b_ndepths = np.concatenate(all_pass1b_ndepths)
    
    # print(len(all_pass1b_correctness))
    # np.savetxt(f"{result_dir}/passn_{config_name}.txt", all_passn_correctness)
    np.savetxt(f"{result_dir}/pass1b_{config_name}.txt", all_pass1b_correctness)
    np.savetxt(f"{result_dir}/naive1b_{config_name}.txt", all_naive1b_correctness)
    np.savetxt(f"{result_dir}/weighted1b_{config_name}.txt", all_weighted1b_correctness)
    np.savetxt(f"{result_dir}/maj1b_{config_name}.txt", all_maj1b_correctness)
    
    # passn_correctness_mean = np.mean(all_passn_correctness)
    pass1b_correctness_mean = np.mean(all_pass1b_correctness)
    naive1b_correctness_mean = np.mean(all_naive1b_correctness)
    weighted1b_correctness_mean = np.mean(all_weighted1b_correctness)
    maj1b_correctness_mean = np.mean(all_maj1b_correctness)

    nsamples = len(all_pass1b_correctness)
    # passn_correctness_std = np.std(all_passn_correctness, ddof=1)/np.sqrt(num_trials*num_questions)  
    pass1b_correctness_std = np.std(all_pass1b_correctness, ddof=1)
    pass1b_correctness_std = np.std(all_pass1b_correctness, ddof=1)/np.sqrt(nsamples)
    naive1b_correctness_std = np.std(all_naive1b_correctness, ddof=1)/np.sqrt(nsamples)
    weighted1b_correctness_std = np.std(all_weighted1b_correctness, ddof=1)/np.sqrt(nsamples)
    maj1b_correctness_std = np.std(all_maj1b_correctness, ddof=1)/np.sqrt(nsamples)

    pass1b_ncomps_mean = np.mean(all_pass1b_ncomps)
    pass1b_lengths_mean = np.mean(all_pass1b_lengths)
    pass1b_ncomps_std = np.std(all_pass1b_ncomps, ddof=1)/np.sqrt(nsamples)
    pass1b_lengths_std = np.std(all_pass1b_lengths, ddof=1)/np.sqrt(nsamples)

    pass1b_nphases_mean = np.mean(all_pass1b_nphases)
    pass1b_ndepths_mean = np.mean(all_pass1b_ndepths)
    pass1b_nphases_std = np.std(all_pass1b_nphases, ddof=1)/np.sqrt(nsamples)
    pass1b_ndepths_std = np.std(all_pass1b_ndepths, ddof=1)/np.sqrt(nsamples)

    print(
        f"{pass1b_correctness_mean:0.4f} (\u00B1{pass1b_correctness_std:0.4f}), "
        f"{naive1b_correctness_mean:0.4f} (\u00B1{naive1b_correctness_std:0.4f}),"
        f"{weighted1b_correctness_mean:0.4f} (\u00B1{weighted1b_correctness_std:0.4f}), "
        f"{maj1b_correctness_mean:0.4f} (\u00B1{maj1b_correctness_std:0.4f}), "
        f"{pass1b_ncomps_mean:0.1f}    (\u00B1{pass1b_ncomps_std:0.1f}), "
        f"{pass1b_lengths_mean:0.1f}    (\u00B1{pass1b_lengths_std:0.1f}), "
        f"{pass1b_nphases_mean:0.1f}    (\u00B1{pass1b_nphases_std:0.1f}), "
        f"{pass1b_ndepths_mean:0.1f}    (\u00B1{pass1b_ndepths_std:0.1f})"
    )