from transformers import GPT2TokenizerFast
from transformers import  AutoTokenizer
from openai import OpenAI
import os
import json
from tqdm import tqdm
from huggingface_hub import InferenceClient
from vllm import LLM, SamplingParams
import inflect
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-qa", "--model", dest="qa_model")     
parser.add_argument("--load_data_path", dest="load_path", help='path to load verbalized facts')
parser.add_argument("--openai_key", dest="openai_key")
parser.add_argument("--flant5_tgi", default='http://localhost:3000', dest="TGI_address",help="address for TGI")
parser.add_argument("--dataset", default='webqsp', dest="dataset",help="webqsp or mintaka")
parser.add_argument("--retrieve_mode", default="random", dest="retrieve_mode", help='random or popular or mpnet')
parser.add_argument("--limit", type=int, default=400, dest="token_limit")

   
args = parser.parse_args()

p = inflect.engine()

def check(answers, response):
    correct = 0
    for answer in answers:
        if answer in response:
            correct = 1
            break
    return correct

def trim_with_limit(prompt_wo_info, info):
    prompt_wo_info_length = len(tokenizer.encode(prompt_wo_info, add_special_tokens=False))
    info_limit = max(0,args.token_limit - prompt_wo_info_length)
    return_info = tokenizer.decode(tokenizer.encode(info, add_special_tokens=False)[:info_limit])
    assert(len(tokenizer.encode(info, add_special_tokens=False)[:info_limit]) + prompt_wo_info_length <= args.token_limit)
    return return_info

def webqsp_template(mode,infer):
    prompt = ''
    if mode == 'no_knowledge':
        prompt = "You are a student who have to solve the question. ## Question: {}  ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer['question'])
    elif 'KAPING' in mode:
        prompt_wo_info = "You are a student who have to solve the question. I'll give you fact triples as a context. But if it is not useful, just ignore it and generate your own guess.  ## fact triples:  ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(infer['question'])
        info = trim_with_limit(prompt_wo_info, infer[mode].strip())
        prompt = "You are a student who have to solve the question. I'll give you fact triples as a context. But if it is not useful, just ignore it and generate your own guess.  ## fact triples: {} ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(info,infer['question'])
    elif 'EFSum' in mode or 'KG2Text' in mode:
        prompt_wo_info = "You are a student who have to solve the question. I'll give you a summary as a context. But if it is not useful, just ignore it and generate your own guess.  ## summary:  ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(infer['question'])
        info = trim_with_limit(prompt_wo_info, infer[mode].strip())
        prompt = "You are a student who have to solve the question. I'll give you a summary as a context. But if it is not useful, just ignore it and generate your own guess.  ## summary: {} ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(info, infer['question'])
    elif 'Rewrite' in mode:
        prompt_wo_info = "You are a student who have to solve the question. I'll give you linearly verbalized triples as a context. But if it is not useful, just ignore it and generate your own guess.  ## linearly verbalized triples:  ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(infer['question'])
        info = trim_with_limit(prompt_wo_info, infer[mode].strip())
        prompt = "You are a student who have to solve the question. I'll give you linearly verbalized triples as a context. But if it is not useful, just ignore it and generate your own guess.  ## linearly verbalized triples: {} ## Question: {} ## You are aware of the answer. Generate only short answer(You have to guess something): ".format(info, infer['question'])
    return prompt

def mintaka_template(mode,infer):
    prompt = ''
    if mode == 'no_knowledge':
        prompt = "You are a student who have to solve the question. ## Question: {}  ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer['question'])
    elif 'KAPING' in mode:
        prompt_wo_info = "You are a student who have to solve the question. You should answer according to given triples, if it is relevant. ## triples:  ## Question: {} ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer['question'])
        info = trim_with_limit(prompt_wo_info, infer[mode].strip())
        prompt = "You are a student who have to solve the question. You should answer according to given triples, if it is relevant. ## triples: {} ## Question: {} ## You are aware of the answer. Generate the short answer(You must guess something): ".format(info, infer['question'])
    elif 'EFSum' in mode or 'KG2Text' in mode:
        prompt_wo_info = "You are a student who have to solve the question. You should answer according to given summary, if it is relevant. ## Summary:  ## Question: {} ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer['question'])
        info = trim_with_limit(prompt_wo_info, infer[mode].strip())
        prompt = "You are a student who have to solve the question. You should answer according to given summary, if it is relevant. ## Summary: {} ## Question: {} ## You are aware of the answer. Generate the short answer(You must guess something): ".format(info, infer['question'])
    elif 'Rewrite' in mode:
        prompt_wo_info = "You are a student who have to solve the question. You should answer according to given facts, if it is relevant. ## Facts:  ## Question: {} ## You are aware of the answer. Generate the short answer(You must guess something): ".format(infer['question'])
        info = trim_with_limit(prompt_wo_info, infer[mode].strip())
        prompt = "You are a student who have to solve the question. You should answer according to given facts, if it is relevant. ## Facts: {} ## Question: {} ## You are aware of the answer. Generate the short answer(You must guess something): ".format(info, infer['question'])
    return prompt

def log_result(mode, index, result):
    with open(save_path.format(mode, index), 'w') as f:
        save_data = json.dumps(result)
        f.write(save_data)
        
def lowercase(text):
    if type(text) == list:
        return [t.lower() for t in text]
    elif type(text) == str:
        return text.lower()

def inference(mode,infer):
    result = {}
    if args.dataset == 'webqsp':
        prompt = webqsp_template(mode, infer)
    elif args.datset == 'mintaka':
        prompt = mintaka_template(mode, infer)
        
    if args.qa_model == 'llama':
        output = llm.generate(prompt,sampling_params,use_tqdm=False)
        response = output[0].outputs[0].text
    elif args.qa_model == 'flant5':
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
    result['question'] = infer['question']
    result['answer'] = infer['answer']
    result['prompt'] = prompt
    result['response'] = response
    result['correct'] = check(lowercase(infer['answer']), lowercase(response))
    log_result(mode,index,result)
    return result

def get_mode(data):
    mode_candidates = ['no_knowledge','KAPING','KG2Text','Rewrite','EFSum']
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
    
save_path = f'./{args.dataset}/{args.qa_model}/{{}}/inference_{{}}.json'

with open(args.load_path, 'r') as f:
    data = json.load(f)['data']
mode_list = get_mode(data)
for mode in mode_list:
    os.makedirs(f'./{args.dataset}/{args.qa_model}/{{}}'.format(mode),exist_ok=True)
    
report = {}
for mode in mode_list:
    report[mode] = {'acc': 0}
    
if args.qa_model == 'llama':
    sampling_params = SamplingParams(temperature=0,max_tokens=500, prompt_logprobs=1)
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True, tensor_parallel_size=4, seed=42)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
elif args.qa_model == 'flant5':
    client = InferenceClient(model=args.TGI_address)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
elif args.qa_model == 'gpt':
    client = OpenAI(api_key = args.openai_key)
    tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo')
    
for index, _ in tqdm(enumerate(data), desc='Processing', total=len(data)):
    inference_dict = {}
    infer = sample(index, data)
    for mode in mode_list:
        try:
            inference_dict[mode] = inference(mode, infer)
        except:
            inference_dict[mode] = {}
            inference_dict[mode]['correct'] = 0
            print('error!!')
    for mode in mode_list:
        report[mode]['acc'] += inference_dict[mode]['correct']
    break

for mode in mode_list:
    report[mode]['acc'] = round(report[mode]['acc']/len(data),3)

with open(f'{args.dataset}_{args.qa_model}_{args.retrieve_mode}_accuracy_log.json','w') as f:
    json.dump(report, f)
