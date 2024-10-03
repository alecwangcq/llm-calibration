import os
import json
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Load the MMLU dataset
dataset = load_dataset('cais/mmlu')

# Initialize the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Ensure you're using the correct model path

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model_name=model_name)

# Output file to record the likelihoods
output_file = "likelihoods.jsonl"

# Define the sampling parameters for VLLM
gen_params = SamplingParams(
    temperature=1.0,  # Deterministic output
    top_p=1.0,
    max_tokens=0,  # We don't want to generate new tokens, just evaluate the inputs
    logprobs=True,  # Request log probabilities
)

with open(output_file, 'w') as f_out:
    for example in tqdm(dataset):
        question = example['question']
        choices = example['choices']
        correct_choice = example['answer']

        likelihoods = {}
        for choice_label, choice_text in choices.items():
            prompt = f"{question}\nAnswer: {choice_text}"

            # Generate using VLLM and obtain log probabilities
            outputs = llm.generate(
                [prompt],
                sampling_params=gen_params
            )

            # Extract the log probabilities
            log_likelihood = 0.0
            for output in outputs:
                tokens = output.tokens
                token_logprobs = output.token_logprobs  # Log probabilities of the tokens

                # Sum the log probabilities of the tokens in the prompt
                # Exclude the first token as its probability is not meaningful
                if token_logprobs is not None:
                    log_likelihood = sum(token_logprobs[1:])

            likelihoods[choice_label] = log_likelihood

        # Record the results
        result = {
            "question": question,
            "likelihoods": likelihoods,
            "correct_answer": correct_choice
        }
        f_out.write(json.dumps(result) + '\n')