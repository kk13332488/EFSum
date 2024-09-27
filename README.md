## Requirements
* qlora
* vllm
* transformers
* numpy
* torch
* openai
* tqdm
* deepspeed
* text-generation-inference (https://huggingface.co/docs/text-generation-inference/installation)

## SFT(Supervised Fine-tuning)
To run SFT training with reference summary, run the code below. 
```sh
sh qlora/finetune_efsum.sh WANDB_ENTITY WANDB_PROJECT
```
You can find training data from below link.

https://drive.google.com/drive/folders/1lR29DjCNebgtu3USYcN3Nm8GoNlonZOI?usp=drive_link

Just download the training data you want to use, and put the training data to **qlora/data/**. You have to turn the training file name to **train.json**.

Before training, you have to set your wandb configuration(**WANDB_ENTITY**, **WANDB_PRORJECT**) to the shell script to monitor the training process. if you want to train on the multiple-gpu, check num_processes in accelerate_config.yaml

## Merge SFT Peft adapter
After the sft training, you have to merge the peft adapter to your base model. You can merge the peft adapter and the base model by running the code below. 
```sh
sh qlora/merge-efsum.sh your_sft_adapter_checkpoint
```
You can add your sft adapter checkpoint to **your_sft_adapter_checkpoint**.

## DPO dataset preprocessing
To prepare DPO dataset, the data have to go throught some filtering process. If you want to skip this process, you can use our training dataset from below link.

https://drive.google.com/drive/folders/13VLkOk80YHYr5NzgbYOimHS-DXFmWGlt?usp=drive_link

You have to put the train, eval dataset to **dpo/data** and the name of each file should be **train.json**, **eval.json**. 
To construct the preprocessed dataset, you can check following instructions.

1. SFT data generation

   
Before you get into DPO dataset preprocessing pipeline, you have to generate 5 reference summary with your SFT checkpoint. First, you have to fill the blacks in the **vllm/server_config.yaml**. Then run the code below. After generating 5 reference summary, you have to move the whole reference summary directories from **vllm/results** to **sft_summary/dataset/qa_model** (e.g. sft_summary/mintaka/flan). You can undergo this process by locating **train.json** to *vllm/train_data*. You can download the data from the link below.

https://drive.google.com/drive/folders/10gG6J46nl-LaEye9VziI9j2CyZ-4KRpe?usp=drive_link

```sh
sh vllm/run.sh
```

2. Accuracy Checking

   
This code marks for each summary and write the result. Revise and run this code. You should add _--flant5_tgi_ for FlanT5, _--openai_key_ for GPT.
```python
python paraphrased_SFT_inference/mark_paraphrase_or_SFT_summary.py --dataset mintaka --qa_model flan --load_data_path path/to/parent/directory/of/_sample --save_data_path path/to/save/result/file --flant5_tgi http://localhost:your_port --summary_type SFT
```

3. Hallucination Checking

   
You have to check whether there is a hallucination or not in each summary. Modify and run the code below.
```python 
python hallucination_check/hallucination_checking.py --dataset_name dataset_name --qa_model qa_model --mode reference --openai_key openai_key --K K
```

4. SFT output filtering

   
After accuracy checking and hallucination checking, you can filter out some un-useful summaries from original dataset. Just modify and run the code below.
```python
python paraphrasing/sft_filtering.py --dataset_name dataset_name --qa_model qa_model --K K
```

5. Paraphrasing

   
You can now paraphrase the useful sft summaries. just run the code below. It will simultaneously generate the paraphrased summaries and check the accuracy of each paraphrased summary. But beware that it does not provide the hallucination checking. You have to check hallucination of each summary like what you did in instruction 2. 
```python
python paraphrasing/paraphrasing.py --dataset_name dataset_name --qa_model qa_model --K K --openai_key openai_key
```

6. dpo_filtering

   
Based on the hallucination and accuracy of each paraphrased summary, you can finally generate the DPO training set. Below code will generate the DPO training dataset to the location **dpo/data**. 
```python
python paraphrasing/dpo_filtering.py --split split --dataset_name dataset_name --qa_model qa_model --K K
```

## DPO Training
To run dpo training with the paraphrased summary, run the code below. 
```sh
sh dpo/run_dpo.sh WANDB_ENTITY, WANDB_PROJECT, your_sft_checkpoint
```
You should add your sft model path to 'your_sft_checkpoint'. Then you can refine your sft model with dpo training. 

## Merge DPO Peft apapter
After training, you have to merge the trained dpo adapter to your sft model. To merge the adapter and your sft model, just run the code below.
```python
python dpo/merge_dpo_adapter.py -b your_base_model_path -p peft_model_path
```
You add the base model path to **your_base_model_path**, and you can add your dpo-trained peft model path to **peft_model_path**. Then the merged model will be appeared at the location *EFSum/dpo/checkpoints*

## Summary Generation
After you get your own DPO checkpoints, it's time to generate summaries from various baselines. Experiment settings can be grouped to Dense Evidence(RQ1), Clear Evidence(RQ2). Check _summary/retriever_dense_verbalized_facts.py_ or _summary/clear_verbalized_facts.py_ respectively. Note that you should check https://github.com/UKPLab/plms-graph2text, if you need KG2Text summaries.
We provide a sample shell script. Revise and run the below code.
```sh
sh webqsp_verbalized_facts.sh
```
## Inference
After you get summaries, the only one left is to inference. Check _inference/retriever_and_dense_inference.py_ or _inference/clear_webqsp_inference.sh_. Before you run any inference code, you should export a huggingface cache directory. If you're going to do inference on FlanT5, you should run text-generation-launcher(TGI) in advance. For example,
```
text-generation-launcher --model-id google/flan-t5-xl --max-input-length 3500 --max-total-tokens 4096
```
We provide a sample shell script. Revise and run the below code.
```sh
sh webqsp_inference.sh
```

