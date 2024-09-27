import json
import os
import json
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphrased_result_path", default="EFSum/data/paraphrased_summary/{}/{}/{}_sample/seed_try.json")
    parser.add_argument("--paraphrased_acc_path", default="EFSum/data/paraphrased_summary_acc/{}/{}/{}.json")
    parser.add_argument("--paraphrased_hal_path", default="EFSum/data/paraphrased_summary_hal/{}/hal.json")
    parser.add_argument("--sft_result_path", default="EFSum/data/sft_summary/{}/{}_sample/seed_try.json")
    parser.add_argument("--sft_acc_path", default="EFSum/data/sft_summary_acc/{}/{}/{}.json")
    parser.add_argument("--sft_hal_path", default="EFSum/data/sft_summary_hal/{}/hal.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--dataset_name", type=str, choices=['webqsp', 'mintaka'])
    parser.add_argument("--qa_model", type=str, choices=['chatgpt', 'flan', 'llama'])
    parser.add_argument("--mode", default="reference", choices=['reference', 'paraphrasing'])
    parser.add_argument("--K", type=str)
    parser.add_argument("--openai_key", type=str)
    return parser.parse_args()

def make_prompt(data, args):
    triples = data['one_hop_pruned_all'][:int(args.K)]
    if args.mode == 'reference':
        sft_summary = data['prediction_K{}'.format(args.K)]
    elif args.mode == 'paraphrasing':
        para_summary = data['paraphrased_K{}'.format(args.K)]
    if args.mode == 'reference':
        prompt =  '''You will be given one summary written to provide useful contexts by given source triples from knowledge graphs.
                    Your task is to check whether the given summary induces factual inconsistency.
                    Please make sure you read and understand these instructions carefully. Please keep this evaluation creteria open while reviewing, and refer to it as needed.
                    
                    Evaluation Criteria:
                    Factual Inconsistency (0 or 1): Does the summary untruthful or misleading facts that are not supported by the source triples? If does, mark 1. Otherwise, mark 0.
                    Evaluation Steps:
                    1. read and understand the source triples first. note the entities that are in focus and the relations between them.
                    2. proceed to read through the summary provided.
                    3. compare the information in the summary with that in the source triples. pay particular attention to the entities, actions, and relations.
                    4. mark '1' if the summary contains factual inconsistencies, i.e., if it states untruthful or misleading facts that are not supported by the source triples.
                    5. mark '0' if the summary is consistent with the source triples and does not misrepresent the facts provided by the source triples.

                    remember, you are not assessing the quality of the writing, but the factual consistency of the summary compared to the source triples. perfection in grammar or style does not account for factual consistency. conversely, poor grammar or style does not necessarily mean factual inconsistency. the key lies in the alignment of facts between the source triples and the summary.
                    Factual Inconsistency (0 or 1): Does the summary untruthful or misleading facts that are not supported by the source triples? If does, mark 1. Otherwise, mark 0.
                    
                    Source Triples:
                    {}
                    Summary:
                    {}
                    Does the summary contain factual inconsistency?
                    Answer:'''
        prompt = prompt.format(triples, sft_summary)
    if args.mode == 'paraphrasing':
        prompt =  '''You will be given one summary written to provide useful contexts by given source triples from knowledge graphs.
                    Your task is to check whether the given summary induces factual inconsistency.
                    Please make sure you read and understand these instructions carefully. Please keep this evaluation creteria open while reviewing, and refer to it as needed.
                    
                    Evaluation Criteria:
                    Factual Inconsistency (0 or 1): Does the summary untruthful or misleading facts that are not supported by the source triples? If does, mark 1. Otherwise, mark 0.
                    Evaluation Steps:
                    1. read and understand the source triples first. note the entities that are in focus and the relations between them.
                    2. proceed to read through the summary provided.
                    3. compare the information in the summary with that in the source triples. pay particular attention to the entities, actions, and relations.
                    4. mark '1' if the summary contains factual inconsistencies, i.e., if it states untruthful or misleading facts that are not supported by the source triples.
                    5. mark '0' if the summary is consistent with the source triples and does not misrepresent the facts provided by the source triples.

                    remember, you are not assessing the quality of the writing, but the factual consistency of the summary compared to the source triples. perfection in grammar or style does not account for factual consistency. conversely, poor grammar or style does not necessarily mean factual inconsistency. the key lies in the alignment of facts between the source triples and the summary.
                    Factual Inconsistency (0 or 1): Does the summary untruthful or misleading facts that are not supported by the source triples? If does, mark 1. Otherwise, mark 0.
                    
                    Source Triples:
                    {}
                    Summary:
                    {}
                    Does the summary contain factual inconsistency?
                    Answer:'''
        prompt = prompt.format(triples, para_summary)
    
    return prompt

def main(args):
    client = OpenAI(api_key=args.openai_key)


    data_to_save = {}
    
    data_to_save_tmp = []
    for sample_i in range(5):
        if args.mode == 'reference':
            print('loading {} data...'.format(sample_i))
            with open(args.sft_result_path.format(args.dataset_name, sample_i+1), 'r') as f:
                total_data = json.load(f)['data']
        elif args.mode == 'paraphrasing':
            with open(args.paraphrased_result_path.format(args.dataset_name, args.qa_model, sample_i+1), 'r') as f:
                total_data = json.load(f)['data']
        print('start hallucination checking!')
        total_len = len(total_data)
        acc_list = []
        for i in tqdm(range(total_len)):
            data = total_data[i]

            message = make_prompt(data, args)
        
            messages = [{'role':'user', 'content':message}]

            completion = client.chat.completions.create(
            model = 'gpt-4',
            messages=messages,
            temperature=0.1,
            seed=42
            )

            chat_response = completion.choices[0].message.content
            chat_response = chat_response.lower()
            chat_response = chat_response.strip("'\"")
            
            acc_list.append(chat_response)
        data_to_save_tmp.append(acc_list)

    data_to_save_tmp = np.array(data_to_save_tmp).T.tolist()
    for index, save_list in enumerate(data_to_save_tmp):
        data_to_save[index] = save_list

    with open(args.sft_hal_path.format(args.dataset_name), 'w') as f:
        data = json.dumps(data_to_save)
        f.write(data)
    print('saved!')

if __name__ == "__main__":
    args = parse_args()
    main(args)