import os

os.environ["WANDB_PROJECT"] = "data-distil"

# # OPTIONAL: set the HF_HOME environment variable to the directory where the Hugging Face models will be stored, same for wandb logs
# os.environ["HF_HOME"] = "."
# os.environ["WANDB_DIR"] = "."


from dotenv import load_dotenv
import pickle

load_dotenv()  # take environment variables from .env
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError(
        "Hugging Face token not found in environment variables. Please set HF_AUTH_TOKEN."
    )

os.environ["HF_TOKEN"] = token

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
import torch
import wandb
import numpy as np

assert torch.cuda.is_available()
device = torch.device("cuda")

import sys

# Add the directory containing 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.prompt_utils import create_prompt_mistral_harness
from constants import PAD_TOKEN, MAX_LENGTH
import datetime
import random
from typing import Callable

# To support finetuning on answer tokens only
RESPONSE_TOKENS = {
    "mistralai/Mistral-7B-Instruct-v0.3": [781, 3588, 17749, 29515],
    "meta-llama/Meta-Llama-3-8B-Instruct": [16533, 25],
}


# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def f_tokenize_chat(row, tokenizer, f_prompt):
    prompt = f_prompt(row)
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    return tokenizer(
        prompt,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

def load_and_tokenize_dataset(
    dataset_id: str,
    model_id: str,
    tokenizer,
    f_prompt: Callable,
    n_val: int,
    n_train: int,
    ds_level: int,
    seed: int,
):

    if dataset_id == "openai/gsm8k":

        dataset = load_dataset("openai/gsm8k", "main")
        dataset_train = dataset["train"]
        dataset_test = dataset["test"]

    elif dataset_id == "microsoft/orca-math-word-problems-200k":

        dataset = load_dataset("microsoft/orca-math-word-problems-200k", "default")
        dataset_train = dataset["train"]

        # TODO: change to MATH when finetuning for MATH dataset
        dataset = load_dataset("openai/gsm8k", "main")
        # dataset_test = dataset["test"].shuffle(seed=seed).select(list(range(n_val)))

        dataset_test = dataset["test"]
        if ds_level:
            dataset_test = dataset_test.filter(
                lambda x: x["level"] == f"Level {ds_level}"
            )

    elif dataset_id == "hendrycks/competition_math":
        dataset = load_dataset("hendrycks/competition_math", "main")
        dataset_train = dataset["train"]

        dataset_test = dataset["test"]
        if ds_level:
            dataset_test = dataset_test.filter(
                lambda x: x["level"] == f"Level {ds_level}"
            )
            dataset_train = dataset_train.filter(
                lambda x: x["level"] == f"Level {ds_level}"
            )

    elif (
        ("evokd" in dataset_id) or ("lion" in dataset_id) or ("orca_math" in dataset_id) or ("self-consistency")
    ):
        # elif "lion" in dataset_id:
        dataset_train = load_from_disk(dataset_id)
        dataset_train = dataset_train["train"]

        if "gsm8k" in dataset_id:
            dataset_test = load_dataset("openai/gsm8k", "main")
        elif "competition_math" in dataset_id:
            dataset_test = load_dataset("hendrycks/competition_math", "main")
            # since during synthetic data generation we use 'question' and 'answer' when saving the data
            dataset_train = dataset_train.rename_columns(
                {"question": "problem", "answer": "solution"}
            )
        else:
            raise ValueError(f"Dataset {dataset_id} not supported.")
        dataset_test = dataset_test["test"]

    else:
        raise ValueError(f"Dataset {dataset_id} not supported")

    if n_train and n_train < len(dataset_train):
        dataset_train = dataset_train.shuffle(seed=seed).select(list(range(n_train)))
    print(f"Length of finetuning dataset: {len(dataset_train)}")

    if n_val and n_val < len(dataset_test):
        dataset_test = dataset_test.shuffle(seed=seed).select(list(range(n_val)))
    print(f"Length of validation dataset: {len(dataset_test)}")

    if (
        model_id == "mistralai/Mistral-7B-Instruct-v0.3"
        or model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
    ):
        _f_tokenize = f_tokenize_chat
    else:
        raise ValueError(f"Model {model_id} not supported.")
    f_tokenize = lambda row: _f_tokenize(row, tokenizer, f_prompt)

    train_inputs = dataset_train.map(f_tokenize)
    valid_inputs = dataset_test.map(f_tokenize)

    # only keep 'input_ids' and 'attention_mask'
    train_inputs = train_inputs.select_columns(["input_ids", "attention_mask"])
    valid_inputs = valid_inputs.select_columns(["input_ids", "attention_mask"])

    def to_device(row):
        return {k: v[0] for k, v in row.items()}

    train_inputs = train_inputs.map(to_device)
    valid_inputs = valid_inputs.map(to_device)

    return train_inputs, valid_inputs


def finetune(
    model_id: str,
    f_prompt=create_prompt_mistral_harness,
    dataset_id: str = "openai/gsm8k",
    batch_size: int = 24,
    answer_loss_only: bool = True,
    quantize: bool = True,
    orca_math_size: int = None,
    n_val: int = 250,
    lora_rank: int = 64,
    n_train: int = None,
    ds_level: int = None,
    seed: int = 42,
    n_epochs: int = 2,
    output_root: str = "./lora_output",
):

    set_seed(seed)

    output_dir = f"{output_root}/{model_id}" 
    now = datetime.datetime.now()
    lora_version = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir += f"/{lora_version}"
    print(f"Output directory: {output_dir}")
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1) Load and prepare the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = PAD_TOKEN

    train_inputs, valid_inputs = load_and_tokenize_dataset(
        dataset_id=dataset_id,
        model_id=model_id,
        tokenizer=tokenizer,
        f_prompt=f_prompt,
        n_val=n_val,
        n_train=n_train,
        ds_level=ds_level,
        seed=seed,
        orca_math_size=orca_math_size,
    )

    # 2) Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=quantize,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # 3) Init LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0.05,
        target_modules="all-linear",
    )

    if quantize:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(model)

    # 4) LoRA finetuning
    config = {
        "lora_config": lora_config,
        "learning_rate": 1e-4,
        "num_train_epochs": n_epochs,
        "gradient_accumulation_steps": 2,
        "per_device_train_batch_size": batch_size,
        "gradient_checkpointing": True,
    }

    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # fp16=True,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        eval_steps=10,
        evaluation_strategy="steps",
        # save_strategy="steps",                      # we will automatically save results in the output dir
        # save_steps=100,
        save_strategy="no",
        optim="adamw_torch_fused",
        max_steps=-1,
        report_to="wandb",
        seed=seed,
        data_seed=seed,
        **{k: v for k, v in config.items() if k != "lora_config"},
    )

    if answer_loss_only:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=RESPONSE_TOKENS[model_id],
            tokenizer=tokenizer,
            mlm=False,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Create Trainer instance
    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_inputs,
        eval_dataset=valid_inputs,
        data_collator=data_collator,
        callbacks=[],
    )

    trainer.train()

    model.save_pretrained(output_dir)
    wandb.finish()

    return output_dir, lora_version

