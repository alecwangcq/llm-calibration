#!/bin/bash

for csv_file in test/*_test.csv
do
    echo "Processing ${csv_file}."
    python cali_mmlu_base_llm_o1.py --csv_dir "$csv_file"
done