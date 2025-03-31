from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa
from unsloth.chat_templates import get_chat_template  # noqa

import gc
import logging
import os

import pandas as pd
from datasets import Dataset

# Saving model
from transformers import TrainingArguments
from trl import SFTTrainer

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"{__file__}.log", encoding="utf-8", level=logging.DEBUG)

model_name_or_path = "unsloth/Llama-3.2-1B-bnb-4bit"
max_seq_length = 500

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name_or_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "o_proj",
        "gate_proj",
    ],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",  # type: ignore
    random_state=32,
    loftq_config=None,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

last_checkpoint = None

if os.path.isdir("outputs") and any("checkpoint" in f for f in os.listdir("outputs")):
    last_checkpoint = sorted(
        [f for f in os.listdir("outputs") if "checkpoint" in f],
        key=lambda x: int(x.split("-")[1]),
    )[-1]
    logger.info(f"Resuming taining from checkpoint {last_checkpoint}")

tokenizer = get_chat_template(tokenizer, "llama-3.1")

data_prompt = """Analyze the provided context data about the patient.
Generate a fitting clinical note for the specified category with maximum 500 words.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def format_prompts(examples: pd.DataFrame):
    prompts = examples["prompt"]
    notes = examples["label"]
    texts = []
    for prompt, note in zip(prompts, notes):
        text = data_prompt.format(prompt, note) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


train_dataset = pd.read_csv("./data/train_prompts.csv")
hf_train_dataset = Dataset.from_pandas(train_dataset)

del train_dataset
gc.collect()

hf_train_dataset = hf_train_dataset.map(format_prompts, batched=True)
hf_train_dataset = hf_train_dataset.remove_columns("label")

eval_dataset = pd.read_csv("./data/val_prompts.csv")
hf_eval_dataset = Dataset.from_pandas(eval_dataset)

del eval_dataset
gc.collect()

hf_eval_dataset = hf_eval_dataset.map(format_prompts, batched=True)
hf_eval_dataset = hf_eval_dataset.remove_columns("label")

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_train_dataset,
    eval_dataset=hf_eval_dataset,
    processing_class=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=128,
        warmup_steps=10000,
        num_train_epochs=5,  # Set this for 1 full training run.
        # max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",  # Use this for WandB etc
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=5000,
        eval_steps=5000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        resume_from_checkpoint=last_checkpoint,
    ),
)

trainer_stats = trainer.train()  # type: ignore
