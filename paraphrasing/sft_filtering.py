import json

import json
import argparse
import random


global total_len


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_result_path", default="data/sft_summary/{}/{}_sample/seed_try.json")
    parser.add_argument("--sft_acc_path", default="data/sft_summary_acc/{}/{}/{}.json")
    parser.add_argument("--sft_hal_path", default="data/sft_summary_hal/{}/hal.json")
    parser.add_argument("--dataset_name", type=str, choices=['webqsp', 'mintaka'])
    parser.add_argument("--qa_model", type=str, choices=['chatgpt', 'flan', 'llama'])
    parser.add_argument("--K", type=str)
    return parser.parse_args()

def sft_results(args):

    acc_path = (args.sft_acc_path).format(args.dataset_name, args.qa_model, args.qa_model)
    hal_path = (args.sft_hal_path).format(args.dataset_name)

    with open(acc_path, 'r') as f:
        acc_data = json.load(f)
    with open(hal_path, 'r') as f:
        hal_data = json.load(f)

    sft_summary = [[] for _ in range(total_len)]
    for sample_i in range(5):
        load_path = (args.sft_result_path).format(args.dataset_name, sample_i+1)
        with open(load_path, 'r') as f:
            total_data = json.load(f)['data']
        for i in range(total_len):
            data = total_data[i]
            if len(data) != 0:
                sft_summary[i].append(data['prediction_K{}'.format(args.K)])
                # sft_accs[i].append(acc_data[str(i)][sample_i])
                # sft_hals[i].append(hal_data[str(i)][sample_i])
            else:
                sft_summary[i].append(0)
                # sft_accs[i].append(0)
                # sft_accs[i].append(1)
    sft_accs = []
    sft_hals = []
    for i in range(total_len):
        sft_accs.append(acc_data[str(i)])
        sft_hals.append(hal_data[str(i)])

    
    return sft_summary, sft_accs, sft_hals


def main(args):
    if args.dataset_name == 'webqsp':
        with open('data/sft_summary/webqsp/1_sample/seed_try.json', 'r') as f:
            total_data = json.load(f)['data']
    elif args.dataset_name == 'mintaka':
        with open('data/sft_summary/mintaka/1_sample/seed_try.json', 'r') as f:
            total_data = json.load(f)['data']

    global total_len
    total_len = len(total_data)

    # sft_summary, sft_accs, sft_hals = sft_results(sft_path, sft_acc_path, sft_hal_path)
    sft_summary, sft_accs, sft_hals = sft_results(args)

    data_to_save = {}
    for i in range(total_len):
        data = total_data[i]

        data_to_save_iter = []
        # print(sft_accs[i], sft_hals[i])
        if sft_accs[i] != []:
            if 1 in sft_accs[i] and 0 in sft_hals[i]:
                acc_ones = []
                acc_hal_ones = []
                for acc_i, acc in enumerate(sft_accs[i]):
                    if acc == 1:
                        acc_ones.append(acc_i)
                for acc_one in acc_ones:
                    if sft_hals[i][acc_one] == 0:
                        acc_hal_ones.append(acc_one)
                
                if len(acc_hal_ones) != 0:
                    for acc_hal_i in range(5):
                        if acc_hal_i in acc_hal_ones:
                            data_to_save_iter.append(1)
                        else:
                            data_to_save_iter.append(0)
        print(data_to_save_iter)
        data_to_save[str(i)] = data_to_save_iter
    
    with open('data/sft_summary_acc_filtered/{}/{}/{}.json'.format(args.dataset_name, args.qa_model, args.qa_model), 'w') as f:
        data = json.dumps(data_to_save)
        f.write(data)
    print('saved!')


    
if __name__ == "__main__":
    args = parse_args()
    main(args)