
from openai import OpenAI
import os
import json
from tqdm import tqdm
from huggingface_hub import InferenceClient
from vllm import LLM, SamplingParams
import inflect
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-qa", "--qa_model", dest="qa_model")     
parser.add_argument("--load_data_path", dest="load_path", help='path to summaries. should be set to the parent directory of {}_sample/')
parser.add_argument("--save_data_path", dest="save_path", help='path to save mark result. the filename is formatted with qa_model.json')
parser.add_argument("--openai_key", dest="openai_key")
parser.add_argument("--flant5_tgi", default='http://localhost:3000', dest="TGI_address",help="address for TGI")
parser.add_argument("--dataset", default='webqsp', dest="dataset",help="webqsp or mintaka")
parser.add_argument("--summary_type", default='paraphrased', dest="summary_type",help="paraphrase or SFT")
   
args = parser.parse_args()

random.seed(42)
p = inflect.engine()

def check(answers, response):
    correct = 0
    for answer in answers:
        if answer in response:
            correct = 1
            break
    return correct

def webqsp_template(mode,infer):
    prompt = ''
    if mode == 'no_knowledge':
        prompt = "You are a student who have to solve the question. ## Question: {}  ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer['question'])
    elif 'KAPING' in mode:
        prompt = "You are a student who have to solve the question. I'll give you fact triples as a context. But if it is not useful, just ignore it and generate your own guess.  ## fact triples: {} ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(infer[mode].strip(),infer['question'])
    elif 'EFSum' in mode or 'KG2Text' in mode:
        prompt = "You are a student who have to solve the question. I'll give you a summary as a context. But if it is not useful, just ignore it and generate your own guess.  ## summary: {} ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(infer[mode].strip(), infer['question'])
    elif 'Rewrite' in mode:
        prompt = "You are a student who have to solve the question. I'll give you linearly verbalized triples as a context. But if it is not useful, just ignore it and generate your own guess.  ## linearly verbalized triples: {} ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(infer[mode].strip(), infer['question'])
    return prompt

def mintaka_template(mode,infer):
    prompt = ''
    if mode == 'no_knowledge':
        prompt = "You are a student who have to solve the question. ## Question: {}  ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer['question'])
    elif 'KAPING' in mode:
        prompt = "You are a student who have to solve the question. You should answer according to given triples, if it is relevant. ## triples: {} ## Question: {} ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer[mode].strip(), infer['question'])
    elif 'EFSum' in mode or 'KG2Text' in mode:
        prompt = "You are a student who have to solve the question. You should answer according to given summary, if it is relevant. ## Summary: {} ## Question: {} ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer[mode].strip(), infer['question'])
    elif 'Rewrite' in mode:
        prompt = "You are a student who have to solve the question. You should answer according to given facts, if it is relevant. ## Facts: {} ## Question: {} ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer[mode].strip(), infer['question'])
    return prompt   
        
def lowercase(text):
    if type(text) == list:
        return [t.lower() for t in text]
    elif type(text) == str:
        return text.lower()

def inference(mode,infer):
    if args.dataset == 'webqsp':
        prompt = webqsp_template(mode, infer)
    elif args.datset == 'mintaka':
        prompt = mintaka_template(mode, infer)
    if args.qa_model == 'llama':
        output = llm.generate(prompt,sampling_params,use_tqdm=False)
        response = output[0].outputs[0].text
    elif args.qa_model == 'flan':
        response = client.text_generation(prompt=prompt, max_new_tokens=400, temperature=0.001,seed=42)
    elif args.qa_model == 'gpt':
        messages = [{'role':'user', 'content':prompt}]
        completion = client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages=messages,
            temperature=0.1,
            seed=42
        )
        response = completion.choices[0].message.content
    return check(lowercase(infer['answer']), lowercase(response))

def get_mode(data):
    mode_candidates = ['paraphrase','SFT']
    mode_list = []
    keys = data[0].keys()
    for mode in mode_candidates:
        for key in keys:
            if mode in key:
                mode_list.append(key)
    return mode_list

def sample(index, data):
    ret_dict = data[index]
    if args.dataset == 'webqsp':
        ret_dict['answer'] = data[index]['answers_str']
        ret_dict['question'] = data[index]['utterance']
    if args.dataset == 'mintaka':
        ret_dict['answerType'] = data[index]['answer']['answerType']
        ret_dict['answer'] = [data[index]['answer']['mention']]
        if ret_dict['answerType'] == 'numerical':
            answer_word = p.number_to_words(ret_dict['answer'][0])
            ret_dict['answer'] += [answer_word]
    return ret_dict

def merge_summaries():
    data = []
    for x in range(1,6):
        with open(os.path.join(args.load_path, f'{x}_sample', 'seed_try.json'), 'r') as f:
            data.append(json.load(f)['data'])
    if args.dataset == 'webqsp':
        K = 30
    elif args.dataset == 'mintaka':
        K = 50
    new_data = []
    empty_list = []
    for index in range(len(data[0])):
        new_dict = data[0][index]
        is_empty = False
        for x in range(1,6):
            try:
                new_dict[args.summary_type + f'_K{x}'] = data[x-1][index][f'prediction_K{K}']
            except:
                empty_list.append(index)
                is_empty = True
                break
        if is_empty:
            new_data.append([])
        else:
            new_data.append(new_dict)
    return empty_list, new_data
        
save_path = f'./{args.dataset}/{args.qa_model}/{{}}/inference_{{}}.json'

empty_list, data = merge_summaries()
mode_list = get_mode(data)
for mode in mode_list:
    os.makedirs(f'./{args.dataset}/{args.qa_model}/{{}}'.format(mode),exist_ok=True)
    
report = {}
for mode in mode_list:
    report[mode] = {'acc': 0}
    
if args.qa_model == 'llama':
    sampling_params = SamplingParams(temperature=0,max_tokens=500, prompt_logprobs=1)
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True, tensor_parallel_size=4, seed=42)
elif args.qa_model == 'flan':
    client = InferenceClient(model=args.TGI_address)
elif args.qa_model == 'gpt':
    client = OpenAI(api_key = args.openai_key)
    
mark_dict = {}

for index, _ in tqdm(enumerate(data), desc='Processing', total=len(data)):
    mark_list = []
    if not(index in empty_list):
        infer = sample(index, data)
        for mode in mode_list:
            try:
                mark_list.append(inference(mode, infer))
            except:
                mark_list.append(0)
                print('error!!')
    mark_dict[index] = mark_list
    
with open(os.path.join(args.save_path, f'{args.qa_model}.json'),'w') as f:
    json.dump(mark_dict, f)
