#!/bin/sh

python ./clear_verbalized_facts.py \
    --load_data_path "../dataset/webqsp_test.json"\
    --save_data_path "../dataset/webqsp_summary.json"\
    --model_path "path/to/distilled/model"\
    --dataset "webqsp"\
    --openai_key "openai_key"