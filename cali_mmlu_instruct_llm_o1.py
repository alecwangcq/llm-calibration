import argparse
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import glob
from tqdm import tqdm

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2 - 6
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure model is in evaluation mode
    model.eval()
    return model, tokenizer

def get_choice_ids(tokenizer):
    choice_ids = []
    choices = [' A', ' B', ' C', ' D']
    for c in choices:
        tokens = tokenizer.encode(c, add_special_tokens=False)
        choice_ids.append(tokens[-1])
    return choice_ids

def get_logits(input_texts, model, tokenizer):
    # Ensure left padding and set pad token
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    # import ipdb; ipdb.set_trace()
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        # truncation=True,
        # max_length=tokenizer.model_max_length
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    return logits, inputs

def get_logits_for_each_ans(logits, choice_ids: List[int]):
    # logits: batch_size x sequence_length x vocab_size
    # Get logits of the last token in the input sequence
    last_logits = logits[:, -1, :]  # Shape: batch_size x vocab_size
    # Gather the logits corresponding to the choice_ids
    choice_logits = last_logits[:, choice_ids]  # Shape: batch_size x num_choices
    return choice_logits

def make_output_path(csv_path):
    return csv_path.replace('.csv', '_logits.csv')

def save_result(df, output_path):
    df.to_csv(output_path, index=False)

def eval(df_path, model, tokenizer, batch_size=8):
    test_df = pd.read_csv(df_path, header=None)
    output_path = make_output_path(df_path)
    # model, tokenizer = build_model(model_name)
    choice_ids = get_choice_ids(tokenizer)

    # Add columns for logits and predictions
    for choice in choices:
        test_df[f'Logit_{choice}'] = 0.0
    test_df['Predicted'] = ''
    test_df['Correct'] = test_df.iloc[:, test_df.shape[1]-1]
    num_correct = 0

    prompts = []
    indices = []
    total_samples = len(test_df)
    for idx in tqdm(range(total_samples)):
        prompt = format_example(test_df, idx, include_answer=False)

        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f"Please answer the question with only A, B, C, D: {prompt}"}], 
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt)
        indices.append(idx)

        # Process in batches
        if len(prompts) == batch_size or idx == total_samples - 1:
            logits, _ = get_logits(prompts, model, tokenizer)
            choice_logits = get_logits_for_each_ans(logits, choice_ids)
            # choice_logits shape: batch_size x num_choices

            # Convert logits to CPU for processing
            choice_logits = choice_logits.cpu().numpy()
            for i, index in enumerate(indices):
                # Update logits in the DataFrame
                for j, choice in enumerate(choices):
                    test_df.at[index, f'Logit_{choice}'] = choice_logits[i, j]
                # Determine the predicted choice
                predicted_choice = choices[choice_logits[i].argmax()]
                test_df.at[index, 'Predicted'] = predicted_choice
                # Check if prediction is correct
                correct_answer = str(test_df.iloc[index, 5]).strip()
                print(f'GT: {correct_answer}, Predicted: {predicted_choice}')
                if predicted_choice == correct_answer:
                    num_correct += 1

            # Reset prompts and indices for the next batch
            prompts = []
            indices = []

            # Optionally, save intermediate results
            if idx % (batch_size * 10) == 0 or idx == total_samples - 1:
                accuracy = num_correct / (idx + 1) * 100
                print(f'Processed {idx + 1}/{total_samples} samples, Accuracy so far: {accuracy:.2f}%')
                save_result(test_df, output_path)

    # Final accuracy
    accuracy = num_correct / total_samples * 100
    print(f'\nFinal Accuracy: {accuracy:.2f}%')
    # Save final results
    save_result(test_df, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-70B-Instruct')
    parser.add_argument('--batch_size', type=int, default=24)
    args = parser.parse_args()

    model, tokenizer = build_model(args.model_name)
    for csv_file in glob.glob(f'{args.csv_dir}/*_test.csv'):
        print(f"Processing {csv_file}.")
        eval(csv_file, model, tokenizer, args.batch_size)