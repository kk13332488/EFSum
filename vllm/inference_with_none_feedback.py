# webqsp density sft script
import argparse
import asyncio
import json
import os
import random
import glob
import yaml
import vllm
import copy
import re
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from transformers.trainer_utils import set_seed

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat, OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datasets import load_dataset, concatenate_datasets
import openai
from evaluate import load


import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_model_name", type=str, required=True)
    parser.add_argument("--generation_model_port", type=int, required=True)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--do_iterate", default="Yes", choices=["Yes", "No"])
    parser.add_argument("--iterate_num", type=int, default=5, help="number of iteration")
    parser.add_argument("--num_try", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--dataset_name", type=str, default='mintaka', choices=['webqsp', 'mintaka'])
    parser.add_argument("--mode", default="inference", choices=['construction', 'inference'])
    parser.add_argument("--save_dir", type=str, default="vllm/results")
    parser.add_argument("--K", type=str)
    return parser.parse_args()


def load_and_prepare_data(args):
    with open(args.dataset_path, 'r') as f:
        data = json.load(f)
    all_model_inputs = data['data']
    return all_model_inputs

def random_knowledge(triples,K):
    return random.sample(triples, k = min(K, len(triples)))

def popular_knowledge(triples,K):
    freq_rel = sorted(triples, key=lambda x: triples.count(x[1]), reverse=True)
    return freq_rel[:min(K, len(triples))]

def make_prompt(args, model_input):
    if args.dataset_name == 'webqsp':
        triples = model_input['one_hop_pruned_all'][:int(args.K)]
        question = model_input['utterance']
    if args.dataset_name == 'mintaka':
        triples = model_input['one_hop_pruned_all'][:int(args.K)]
        question = model_input['question']
        
    triples_str = ''
    for triple in triples:
        triples_str += triple[0]
        triples_str += ' | '
        triples_str += triple[1]
        triples_str += ' | '
        triples_str += triple[2]
        triples_str += ' \n '
    
    prompt = 'You are a knowledge graph summarizer for Question Answering. I will give you "Question", "Fact triples". You should turn triples into *summary*. The *summary* should serve as a context to facilitate QA (Question and Answer) tasks. ##Caution1 : The *summary* should not explicitly mention what the correct answer is. ##Caution2 : The *summary* should only conatin information of the given triples. ## Caution3 : Each triplet is seperated with "\n" and head, relation, tail are provided in head | relation | tail format. \n ## Fact triples: \n {} '.format(triples_str) + '## Question: {}'.format(question) + ' ## Summary:'
    return prompt

async def async_generate(llm, model_input, idx, save_dir, args):
    prompt = make_prompt(args, model_input)
    response = await llm.agenerate(prompts=[prompt])  
    response_text= response.generations[0][0].text
    model_input['EFSum_distill_K{}'.format(args.K)] = response_text
    # model_input['prediction'] = response_text

    return model_input

async def generate_concurrently(all_model_input, start_idx, stop, args):
    if args.mode == 'inference':  ## if feedback model
        llm = OpenAI(
            model_name=args.generation_model_name,
            openai_api_base=f"http://localhost:{args.generation_model_port}/v1",
            openai_api_key="EMPTY",
            max_tokens=700,
            temperature=0.1,
            top_p=0.9,
            frequency_penalty=0.0,
            stop=stop,
            seed=42
        )
    elif args.mode == 'construction':
        llm = OpenAI(
            model_name=args.generation_model_name,
            openai_api_base=f"http://localhost:{args.generation_model_port}/v1",
            openai_api_key="EMPTY",
            max_tokens=700,
            temperature=1,
            top_p=0.9,
            frequency_penalty=0.0,
            stop=stop
        )
    tasks = [
        async_generate(llm, model_input, i + start_idx, args.save_dir, args) for i, model_input in enumerate(all_model_input)
    ]
    return await tqdm_asyncio.gather(*tasks)

async def main(args):
    input_dataset = load_and_prepare_data(args)
    if args.mode == 'construction':
        for i in range(args.num_try):
            all_model_inputs = input_dataset
            save_path = os.path.join(args.save_dir, f"{i+1}_sample")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            all_results = await generate_concurrently(all_model_inputs, 0, None, args)  # generate code
            all_results = {'data' : all_results}
            with open(os.path.join(save_path+"", "seed_try.json"), "w", encoding="UTF-8") as f:
                json.dump(all_results, f, indent=4)
    elif args.mode == 'inference':
        all_model_inputs = input_dataset
        save_path = os.path.join(args.save_dir, f"{1}_sample")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        all_results = await generate_concurrently(all_model_inputs, 0, None, args)  # generate code
        all_results = {'data' : all_results}
        with open(os.path.join(save_path+"", "seed_try.json"), "w", encoding="UTF-8") as f:
            json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    nlp = en_core_web_sm.load()
    asyncio.run(main(args))
    print("Done!")