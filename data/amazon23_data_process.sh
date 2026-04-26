#!/bin/bash

python amazon23_data_process.py \
    --dataset {domain} \
    --metadata_file ../meta_{domain}.jsonl \
    --reviews_file ../{domain}.jsonl \
    --user_k 5 \
    --st_year 2018 \
    --st_month 10 \
    --ed_year 2023 \
    --ed_month 9 \
    --output_path ./Amazon23
