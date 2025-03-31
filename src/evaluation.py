from unsloth import FastLanguageModel  # noqa
import os

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm


# get the last checkpoint

last_checkpoint = sorted(
    [f for f in os.listdir("outputs") if "checkpoint" in f],
    key=lambda x: int(x.split("-")[1]),
    reverse=True,
)[0]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model, tokenizer = FastLanguageModel.from_pretrained(
    "outputs/" + last_checkpoint,
    load_in_4bit=True,
)


model = model.to(device)

FastLanguageModel.for_inference(model)

data_prompt = """Analyze the provided context data about the patient.
Generate a fitting clinical note for the specified category with maximum 500 words.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def format_prompts(examples: pd.DataFrame):
    prompts = examples["prompt"]
    texts = []
    for prompt in prompts:
        text = EOS_TOKEN + data_prompt.format(prompt, "")
        texts.append(text)
    return {
        "text": texts,
    }


test_dataset = pd.read_csv("./data/test_prompts.csv").sample(1000)
hf_test_dataset = Dataset.from_pandas(test_dataset)
hf_test_dataset = hf_test_dataset.map(format_prompts, batched=True)
hf_test_dataset = hf_test_dataset.remove_columns("label")

batch_size = 32
num_batches = len(hf_test_dataset) // batch_size
batches = hf_test_dataset.batch(batch_size)
responses = []
for batch in tqdm(batches):
    inputs = tokenizer(
        batch["text"],  # type: ignore
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=500,
        padding_side="left",
    )
    inputs = inputs.to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
    )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    responses.extend([output.split("### Response:")[1] for output in decoded_outputs])
test_dataset["response"] = responses
test_dataset.to_csv("./data/test_responses.csv", index=False)
