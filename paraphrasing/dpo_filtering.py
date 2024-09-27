import json
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphrased_result_path", default="data/paraphrased_summary/{}/{}/{}_sample/seed_try.json")
    parser.add_argument("--sft_result_path", default="data/sft_summary/{}/{}_sample/seed_try.json")
    parser.add_argument("--paraphrased_acc_path", default="data/paraphrased_summary_acc/{}/{}/{}.json")
    parser.add_argument("--paraphrased_hal_path", default="data/paraphrased_summary_hal/{}/hal.json")
    parser.add_argument("--sft_acc_path", default="data/sft_summary_acc/{}/{}/{}.json")
    parser.add_argument("--sft_hal_path", default="data/sft_summary_hal/{}/hal.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--dataset_name", type=str, choices=['webqsp', 'mintaka'])
    parser.add_argument("--qa_model", type=str, choices=['chatgpt', 'flan', 'llama'])
    parser.add_argument("--K", type=str)
    parser.add_argument("--save_dir", type=str, default="./results")
    return parser.parse_args()

def paraphrased_results(args):
    acc_path = (args.paraphrased_acc_path).format(args.dataset_name, args.qa_model, args.qa_model)
    hal_path = (args.paraphrased_hal_path).format(args.dataset_name)
    with open(acc_path, 'r') as f:
        acc_data = json.load(f)
    
    with open(hal_path, 'r') as f:
        hal_data = json.load(f)

    paraphrased = [[] for _ in range(total_len)]
    paraphrased_accs = []
    paraphrased_hals = []
    for sample_i in range(5):
        load_path = (args.paraphrased_result_path).format(args.dataset_name, args.qa_model, sample_i+1)
        with open(load_path, 'r') as f:
            total_data = json.load(f)['data']
        for i in range(total_len):
            data = total_data[i]
            paraphrased[i].append(data['paraphrased_K{}'.format(args.K)])

    for i in range(total_len):
        if len(acc_data[str(i)]) != 0:
            paraphrased.append(data['paraphrased_K{}'.format(args.K)])
            paraphrased_accs.append(acc_data[str(i)])
            paraphrased_hals.append(hal_data[str(i)])
        else:
            paraphrased.append([])
            paraphrased_accs.append([])
            paraphrased_hals.append([])
        
    
    return paraphrased, paraphrased_accs, paraphrased_hals

def sft_results(args):
    acc_path = (args.sft_acc_path).format(args.dataset_name, args.qa_model, args.qa_model)
    hal_path = (args.sft_hal_path).format(args.dataset_name)

    with open(acc_path, 'r') as f:
        acc_data = json.load(f)
    with open(hal_path, 'r') as f:
        hal_data = json.load(f)

    sft_summary = [[] for _ in range(total_len)]
    sft_accs = [[] for _ in range(total_len)]
    sft_hals = [[] for _ in range(total_len)]
    for sample_i in range(5):
        load_path = (args.sft_result_path).format(args.dataset_name, sample_i+1)
        with open(load_path, 'r') as f:
            total_data = json.load(f)['data']
        for i in range(total_len):
            data = total_data[i]
            if len(data) != 0:
                sft_summary[i].append(data['prediction_K{}'.format(args.K)])
                sft_accs[i].append(acc_data[str(i)])
                sft_hals[i].append(hal_data[str(i)])
            else:
                sft_summary[i].append(0)
                sft_accs[i].append(0)
                sft_accs[i].append(1)
    
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

    chosen_summary, chosen_accs, chosen_hals = paraphrased_results(args)
    rejected_summary, rejected_accs, rejected_hals = sft_results(args)
    
    if args.split == 'train':
        start_cnt = 0
        end_cnt = total_len - (total_len)*0.1
    elif args.split == 'eval':
        start_cnt = total_len - (total_len)*0.1
        end_cnt = total_len

    data_to_save = []
    for i in range(start_cnt, end_cnt):
        data = total_data[i]
        if args.dataset_name == 'webqsp':
            question = data['utterance']
            triples = data['one_hop_pruned_all'][:int(args.K)]
        elif args.dataset_name == 'mintaka':
            question = data['question']
            # triples = data['one_hop_pruned_all'][:int(args.K)]
            triples = data['one_hop_pruned'][:int(args.K)]

        triples_str = ''
        for triple in triples:
            triples_str += triple[0]
            triples_str += ' | '
            triples_str += triple[1]
            triples_str += ' | '
            triples_str += triple[2]
            triples_str += ' \n '
        
        prompt = 'You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Fact triples". You should turn triples into *summary*. The *summary* should serve as a context to facilitate QA (Question and Answer) tasks. ##Caution1 : The *summary* should not explicitly mention what the correct answer is. ##Caution2 : The *summary* should only conatin information of the given triples. ## Caution3 : Each triplet is seperated with "\n" and head, relation, tail are provided in head | relation | tail format. \n ## Fact triples: \n {} '.format(triples_str) + '## Question: {}'.format(question) + ' ## Summary:'

        if chosen_accs[i] != []:
            if 1 in chosen_accs[i] and 0 in chosen_hals:
                acc_ones = []
                acc_hal_ones = []
                for acc_i, acc in enumerate(chosen_accs[i]):
                    if acc == 1:
                        acc_ones.append(acc_i)
                for acc_one in acc_ones:
                    if chosen_hals[i][acc_one] == 0:
                        acc_hal_ones.append(acc_one)
                
                if len(acc_hal_ones) != 0:
                    chosen_random_index = random.choice(acc_hal_ones)
                else:
                    continue
            else:
                continue
            
            if 0 in rejected_accs[i]:
                acc_zeros = []
                for acc_i, acc in enumerate(rejected_accs[i]):
                    if acc == 0:
                        acc_zeros.append(acc_i)
                rejected_random_index = random.choice(acc_zeros)
            else:
                continue

            chosen = chosen_summary[i][chosen_random_index].strip()
            rejected = rejected_summary[i][rejected_random_index].strip()

            data_to_save.append({'prompt':prompt, 'chosen':chosen, 'rejected':rejected})
    
    with open('dpo/data/{}.json'.format(args.split), 'w') as f:
        data = json.dumps(data_to_save)
        f.write(data)
    print('saved!')
        

if __name__ == "__main__":
    args = parse_args()
    main(args)