import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_data_path", dest="load_path", help='path to load preprocessed data')
parser.add_argument("--save_data_path", dest="save_path", help='path to save data of verbalized facts')
parser.add_argument("--model_path", dest="model_path", help='path to distilled model')
parser.add_argument("--dataset", default="webqsp", dest="dataset", help='webqsp or mintaka')
parser.add_argument("--openai_key", dest="openai_key")
args = parser.parse_args()

client = OpenAI(api_key = args.openai_key)
sampling_params = SamplingParams(temperature=0.1,max_tokens=1024)
llm = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=4, seed=42)

if args.dataset == 'webqsp':
    question_key = 'utterance'
elif args.dataset == 'mintaka':
    question_key = 'question'
    
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


with open(args.load_path, 'r') as f:
    total_data = json.load(f)['data']

for x in tqdm(total_data, desc="Processing", total=len(total_data)):
    triples = x['one_hop_pruned_all']
    KAPING_txt = KAPING(triples[:150])
    x['KAPING_K10'] = ', '.join(KAPING_txt[:10])
    x['KAPING_K30'] = ', '.join(KAPING_txt[:30])
    x['KAPING_K50'] = ', '.join(KAPING_txt[:50])
    x['KAPING_K70'] = ', '.join(KAPING_txt[:70])
    x['KAPING_K150'] = ', '.join(KAPING_txt[:150])
    rewrite_txt = Rewrite(triples[:10])
    x['Rewrite_K10'] = ' '.join(rewrite_txt[:10])
    x['Rewrite_K30'] = ' '.join(rewrite_txt[:30])
    x['Rewrite_K50'] = ' '.join(rewrite_txt[:50])
    x['Rewrite_K70'] = ' '.join(rewrite_txt[:70])
    x['Rewrite_K150'] = ' '.join(rewrite_txt[:150])
    x['EFSum_prompt_K10'] = EFSum_prompt_summary(x,triples[:10])
    x['EFSum_prompt_K30'] = EFSum_prompt_summary(x,triples[:30])
    x['EFSum_prompt_K50'] = EFSum_prompt_summary(x,triples[:50])
    x['EFSum_prompt_K70'] = EFSum_prompt_summary(x,triples[:70])
    x['EFSum_prompt_K150'] = EFSum_prompt_summary(x,triples[:150])
    x['EFSum_distill_K10'] = EFSum_distill_summary(x,triples[:10])
    x['EFSum_distill_K30'] = EFSum_distill_summary(x,triples[:30])
    x['EFSum_distill_K50'] = EFSum_distill_summary(x,triples[:50])
    x['EFSum_distill_K70'] = EFSum_distill_summary(x,triples[:70])
    x['EFSum_distill_K150'] = EFSum_distill_summary(x,triples[:150])
    
with open(args.save_path, 'w') as f:
    json.dump({'data':total_data},f)