#!/bin/sh

#Llama2
python ./clear_inference.py \
    --load_data_path "../dataset/webqsp_summary.json"\
    --model "llama"\
    --dataset "webqsp"

#FlanT5
#You should run TGI for FlanT5 first.
: <<'END'
python ./clear_inference.py \
    --load_data_path "../dataset/webqsp_summary.json"\
    --model "flant5"\
    --flant5_tgi "tgi_address"\
    --dataset "webqsp"
END

: <<'END'
python ./clear_inference.py \
    --load_data_path "../dataset/webqsp_summary.json"\
    --model "gpt"\
    --openai_key "openai key"\
    --dataset "webqsp"
END

#dense_setting example
: <<'END'
#Llama2
python ./dense_inference.py \
    --load_data_path "../dataset/webqsp_summary_popular.json"\
    --model "llama"\
    --dataset "webqsp"\
    --retrieve_mode "popular"\
    --limit 400
END