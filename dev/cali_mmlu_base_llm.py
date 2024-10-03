import argparse
from types import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

choices = ["A", "B", "C", "D"]
def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def build_model(model_name='meta-llama/Meta-Llama-3-70B'):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype='bf16', device_map='auto').cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure model is in evaluation mode
    model.eval()
    return model, tokenizer


def get_logits(input_texts, model, tokenizer):

    # make sure it's left padding, and the last token is for the answer
    tokenizer.padding_side='left'
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    return logits


def get_logits_for_each_ans(logits, choice_ids: List[int]):
    # logits: batch_size x T x vocab_size
    return logits[..., -1, choice_ids]


def make_output_path(csv_path):
    return csv_path.replace('.csv', '_logits.csv')

def save_result(data, output_path):
    import json
    with open(output_path, 'w') as f:
        f.write(json.dumps(data))

def eval(df_path, model_name):
    test_df = pd.read_csv(df_path, header=None)
    """
        example to extract the prompt for each query:
            prompt_end = format_example(test_df, 1, include_answer=True)

    """
    output_path = make_output_path(df_path)
    results = []
    model, tokenizer = build_model(model_name)
    choice_ids = []
    choices = [' A', ' B', ' C', ' D']
    for c in choices:
        tid = tokenizer(c)['input_ids'][-1]
        choice_ids.append(tid)
    for k, v in zip(choices, choice_ids):
        print(f'The key {k} is mapped to {v}.')

    for i in range(len(test_df)):
        prompt = format_example(test_df, i, include_answer=False)
        logits = get_logits(prompt, model, tokenizer)
        logits_of_choices = get_logits_for_each_ans(logits, choice_ids)
        results.append(logits_of_choices)

        if i % 10 == 0:
            save_result(results, output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str)
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    eval(args.csv_dir, args.model_name)