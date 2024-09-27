import os
import json
import random
import argparse
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphrased_result_path", default="data/paraphrased_summary/{}/{}/{}_sample/seed_try.json")
    parser.add_argument("--paraphrased_acc_path", default="data/paraphrased_summary_acc/{}/{}/{}.json")
    parser.add_argument("--sft_result_path", default="data/sft_summary/{}/{}_sample/seed_try.json")
    parser.add_argument("--sft_filtered_acc_path", default="data/sft_summary_acc_filtered/{}/{}/{}.json")
    parser.add_argument("--dataset_name", type=str, choices=['webqsp', 'mintaka'])
    parser.add_argument("--qa_model", type=str, choices=['chatgpt', 'flan', 'llama'])
    parser.add_argument("--K", type=str)
    parser.add_argument("--openai_key", type=str)
    return parser.parse_args()

def return_chosen_sample(args):
    with open(args.sft_filtered_acc_path.format(args.dataset_name, args.qa_model, args.qa_model), 'r') as f:
        acc_data = json.load(f)
    chosen_index = []
    for sample_i in range(len(acc_data)):
        if acc_data[str(sample_i)] != []:
            acc_ones = []
            for acc_i, acc in enumerate(acc_data[str(sample_i)]):
                if acc == 1:
                    acc_ones.append(acc_i)
            random_choice = random.choice(acc_ones)
            chosen_index.append(random_choice)
        else:
            chosen_index.append(-1)
    summary_list = []
    for sample_i in range(5):
        summary_list_tmp = []
        with open(args.sft_result_path.format(args.dataset_name, sample_i+1), 'r') as f:
            total_data = json.load(f)['data']
        for i in range(len(total_data)):
            if total_data[i] == []:
                summary = total_data[i-1]['prediction_K{}'.format(args.K)]
            else:
                summary = total_data[i]['prediction_K{}'.format(args.K)]
            summary_list_tmp.append(summary)
        summary_list.append(summary_list_tmp)
    
    chosen = []
    for chosen_i, index in enumerate(chosen_index):
        if index != -1:
            chosen.append(summary_list[index][chosen_i])
        else:
            chosen.append(None)
    
    return chosen

def main(args):
    client = OpenAI(api_key=args.openai_key)
    chosen_samples = return_chosen_sample(args)
    print('sample choice complete!')

    with open(args.sft_result_path.format(args.dataset_name, 1), 'r') as f:
        total_data = json.load(f)['data']

    total_len = len(total_data)

    paraphrased_samples = []
    for chosen_sample_i, chosen_sample in enumerate(chosen_samples):
        print(chosen_sample_i)
        data = total_data[chosen_sample_i]
        if chosen_sample != None:
            if args.dataset_name == 'mintaka':
                question = data['question']
                answer = data['answer']['mention']

                if answer.lower() in chosen_sample.lower():
                    message = 'You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Answer", "Knowledge Summary". You should pharaphrase the original "Knowledge Summary". The paraphrased summary should serve as a context to facilitate QA (Question and Answer) tasks. Paraphrase the original "Knowledge Summary" to be more helpful to solve the QA. ## Question: {} ## Answer: {} ## Original Summary: {} ## Paraphrased Summary: '.format(question, answer, chosen_sample)
                else:
                    message = 'You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Knowledge Summary". You should pharaphrase the original "Knowledge Summary". The paraphrased summary should serve as a context to facilitate QA (Question and Answer) tasks. Paraphrase the original "Knowledge Summary" to be more helpful to solve the QA. ## Question: {} ## Original Summary: {} ## Paraphrased Summary: '.format(question, chosen_sample)
                messages = [{'role':'user', 'content':message}]
            elif args.dataset_name == 'webqsp':
                question = data['utterance']
                answers = data['answers_str']
                answer_in = 0
                for answer in answers:
                    if answer.lower() in chosen_sample.lower():
                        if answer_in == 0:
                            answer_in += 1
                if answer_in == 1:
                    message = 'You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Answer", "Knowledge Summary". You should pharaphrase the original "Knowledge Summary". The paraphrased summary should serve as a context to facilitate QA (Question and Answer) tasks. Paraphrase the original "Knowledge Summary" to be more helpful to solve the QA. ## Question: {} ## Answer: {} ## Original Summary: {} ## Paraphrased Summary: '.format(question, answers, chosen_sample)
                elif answer_in == 0:
                    message = 'You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Knowledge Summary". You should pharaphrase the original "Knowledge Summary". The paraphrased summary should serve as a context to facilitate QA (Question and Answer) tasks. Paraphrase the original "Knowledge Summary" to be more helpful to solve the QA. ## Question: {} ## Original Summary: {} ## Paraphrased Summary: '.format(question, chosen_sample)
                messages = [{'role':'user', 'content':message}]

            completion = client.chat.completions.create(
                model = 'gpt-3.5-turbo',
                messages=messages,
                temperature=1,
                n=5
            )

            chat_response0 = completion.choices[0].message.content
            chat_response1 = completion.choices[1].message.content
            chat_response2 = completion.choices[2].message.content
            chat_response3 = completion.choices[3].message.content
            chat_response4 = completion.choices[4].message.content

            chat_response_list = [chat_response0, chat_response1, chat_response2, chat_response3, chat_response4]
            response_acc_list = []
            for response in chat_response_list:
                message = "You are a student who have to solve the question. I'll give you a triples as a context. You should answer according to given summary, if it is relevant. ## Caution: If the answer is number, then generate arabic number, not the english number. ## summary: {} ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(response, question)
                print(message)
                messages = [{'role':'user', 'content':message}]

                completion = client.chat.completions.create(
                model = 'gpt-3.5-turbo',
                messages=messages,
                temperature=0.1,
                seed=42
                )

                chat_response = completion.choices[0].message.content
                chat_response = chat_response.lower()
                chat_response = chat_response.strip("'\"")
                correct = 0
                print(answer)
                print(chat_response)
                print('')
                if args.dataset_name == 'webqsp':
                    added = 0
                    for answer in answers:
                        if answer.lower() in chat_response.lower():
                            if added == 0:
                                added += 1
                                correct += 1

                elif args.dataset_name == 'mintaka':
                    if answer.lower() in chat_response.lower():
                        correct += 1

                response_acc_list.append(correct)

            paraphrased_samples.append([chat_response_list, response_acc_list])
        else:
            paraphrased_samples.append([[],[]])
    try:
        os.mkdir(args.paraphrased_result_path.format(args.dataset_name, args.qa_model, 1).split('seed_try')[0][:-1])
        os.mkdir(args.paraphrased_result_path.format(args.dataset_name, args.qa_model, 2).split('seed_try')[0][:-1])
        os.mkdir(args.paraphrased_result_path.format(args.dataset_name, args.qa_model, 3).split('seed_try')[0][:-1])
        os.mkdir(args.paraphrased_result_path.format(args.dataset_name, args.qa_model, 4).split('seed_try')[0][:-1])
        os.mkdir(args.paraphrased_result_path.format(args.dataset_name, args.qa_model, 5).split('seed_try')[0][:-1])
    except:
        pass
    for i in range(5):
        with open(args.sft_result_path.format(args.dataset_name, i+1), 'r') as f:
            total_data = json.load(f)['data']
        data_to_save = []
        for j in range(total_len):
            data = total_data[i]
            if paraphrased_samples[j][0] != []:
                data['paraphrased_K{}'.format(args.K)] = paraphrased_samples[j][0][i]
            data_to_save.append(data)
        
        data_to_save = {'data':data_to_save}
        with open(args.paraphrased_result_path.format(args.dataset_name, args.qa_model, i+1), 'w') as f:
            data_to_save = json.dumps(data_to_save)
            f.write(data_to_save)
    
    acc_dict = {}
    print(paraphrased_samples)
    for i in range(total_len):
        acc_dict[str(i)] = paraphrased_samples[i][1]
    
    with open(args.paraphrased_acc_path.format(args.dataset_name, args.qa_model, args.qa_model), 'w') as f:
        data_to_save = json.dumps(acc_dict)
        f.write(data_to_save)

    print('saved!')

if __name__ == "__main__":
    args = parse_args()
    main(args)