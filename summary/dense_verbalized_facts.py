import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from collections import Counter
from openai import OpenAI
import random
random.seed(42)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_data_path", dest="load_path", help='path to load preprocessed data')
parser.add_argument("--save_data_path", dest="save_path", help='path to save data of verbalized facts')
parser.add_argument("--model_path", dest="model_path", help='path to distilled model')
parser.add_argument("--dataset", default="webqsp", dest="dataset", help='webqsp or mintaka')
parser.add_argument("--retrieve_mode", default="random", dest="retrieve_mode", help='random or popular or mpnet')
parser.add_argument("--openai_key", dest="openai_key")
args = parser.parse_args()

client = OpenAI(api_key = args.openai_key)
sampling_params = SamplingParams(temperature=0.1,max_tokens=1024)
llm = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=4, seed=42)

if args.dataset == 'webqsp':
    question_key = 'utterance'
elif args.dataset == 'mintaka':
    question_key = 'question'
    
def popular_triples(triples,K):
    frequency_counter = Counter(sublist[1] for sublist in triples)
    sorted_sublists = sorted(triples, key=lambda x: (-frequency_counter[x[1]], x[1]))
    return sorted_sublists[:min(K, len(triples))]

def random_triples(triples,K):
    return random.sample(triples, k = min(K, len(triples)))

def triples_tuple(x):
    ret = f'({x[0]},{x[1]},{x[2]})'
    return ret

def KAPING(triples):
    KAPING_verbalized = []
    for triple in triples:
        KAPING_verbalized.append(triples_tuple(triple))
    return KAPING_verbalized

def KG2Text(triples):
    #check KG2Text repo.
    pass

def Rewrite(triples):
    Rewrite_txt = []
    template = 'Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: {}. The sentence is:'
    for triple in triples:
        triples_str = triples_tuple(triple)
        msg = template.format(triples_str)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": msg}],
            temperature = 0,
            seed=42
        )
        Rewrite_txt.append(response.choices[0].message.content)
    return Rewrite_txt

def triples_bar_separate(triples):
    triples_str = ''
    for triple in triples:
        h, r, t = triple[0], triple[1], triple[2]
        triples_str += h
        triples_str += ' | '
        triples_str += r
        triples_str += ' | '
        triples_str += t
        triples_str += ' \n '
    return triples_str

def prompt_template(triples, data):
    triples_str = triples_bar_separate(triples)
    question = data[question_key]
    message = 'You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Fact triples". You should turn triples into *summary*. The *summary* should serve as a context to facilitate QA (Question and Answer) tasks. ##Caution1 : The *summary* should not explicitly mention what the correct answer is. ##Caution2 : The *summary* should only conatin information of the given triples. ## Caution3 : Each triplet is seperated with "\n" and head, relation, tail are provided in head | relation | tail format.' + ' ## Question: ' + question + ' ' + '## Fact triples: ' + triples_str + '## Summary: '
    return message

def EFSum_prompt_summary(data,triples):
    message = prompt_template(triples, data)
    messages = [{'role':'user', 'content':message}]
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages=messages,
        temperature=0.1,
        seed=42
    )
    chat_response = completion.choices[0].message.content
    return chat_response

def EFSum_distill_summary(data,triples):
    message = prompt_template(triples, data)
    output = llm.generate(message,sampling_params,use_tqdm=False)
    chat_response = output[0].outputs[0].text
    return chat_response

def dense_summary(x, triples, EFSum_mode):
    size = [0,30,60,90]
    ret = ''
    for s in size:
        if len(triples) < s:
            break
        else:
            if EFSum_mode == 'EFSum_prompt_summary':
                ret += EFSum_prompt_summary(x, triples[s:s+30])
            elif EFSum_mode == 'EFSum_distill_summary':
                ret += EFSum_distill_summary(x, triples[s:s+30])
    return ret


with open(args.load_path, 'r') as f:
    total_data = json.load(f)['data']

for x in tqdm(total_data, desc="Processing", total=len(total_data)):
    
    if args.retrieve_mode == 'random':
        triples = x['one_hop']
        triples = random_triples(triples, 120)
    elif args.retrieve_mode == 'popular':
        triples = x['one_hop']
        triples = popular_triples(triples, 120)
    elif args.retrieve_mode == 'mpnet':
        triples = ['one_hop_pruned_all'][:120]
        
    K = 50
        
    KAPING_txt = KAPING(triples[:K])
    x[f'KAPING_{args.retrieve_mode}'] = ', '.join(KAPING_txt)
    rewrite_txt = Rewrite(triples[:K])
    x[f'Rewrite_{args.retrieve_mode}'] = ' '.join(rewrite_txt)
    x[f'EFSum_prompt_{args.retrieve_mode}'] = dense_summary(x,triples, 'EFSum_prompt_summary')
    x[f'EFSum_distill_{args.retrieve_mode}'] = dense_summary(x,triples, 'EFSum_distill_summary')
    break
with open(args.save_path, 'w') as f:
    json.dump({'data':total_data},f)