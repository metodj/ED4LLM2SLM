import os


os.environ["WANDB_PROJECT"] = "data-distil"

# # OPTIONAL: set the HF_HOME environment variable to the directory where the Hugging Face models will be stored, same for wandb logs
# os.environ["HF_HOME"] = "."
# os.environ["WANDB_DIR"] = "."

import gc
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError(
        "Hugging Face token not found in environment variables. Please set HF_AUTH_TOKEN."
    )
os.environ["HF_TOKEN"] = token
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

# Add the directory containing 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finetuning.lora import finetune
from inference.inference import inference
from evaluation.gpt4_eval import evaluate_gpt4
from utils.prompt_utils import (
    create_prompt_mistral_harness,
    create_prompt_test_mistral_harness,
    create_prompt_mistral_harness_math,
    create_prompt_test_mistral_harness_math,
)
from utils.gpt4 import extract_label

import torch

F_PROMPT_DICT = {
    "gsm8k": (create_prompt_mistral_harness, create_prompt_test_mistral_harness),
    "competition_math": (
        create_prompt_mistral_harness_math,
        create_prompt_test_mistral_harness_math,
    ),
}


def pipeline(
    model: str,
    bs_train: int,
    train_ds: str,
    n_train: int,
    test_ds: str,
    bs_test: int,
    n_test: int,
    output_root: str,
    ds_level: int = None,
    answer_loss_only: bool = False,
    lora_version: str = None,
    run_id: int = None,
    n_epochs: int = 2,
):
    if "gsm8k" in test_ds:
        f_prompt, f_prompt_test = F_PROMPT_DICT["gsm8k"]
    elif "competition_math" in test_ds:
        f_prompt, f_prompt_test = F_PROMPT_DICT["competition_math"]
    else:
        raise ValueError(f"Unknown dataset: {test_ds}")

    if train_ds:
        _, lora_version = finetune(
            model,
            f_prompt=f_prompt,
            batch_size=bs_train,
            dataset_id=train_ds,
            n_train=n_train,
            ds_level=ds_level,
            answer_loss_only=answer_loss_only,
            n_epochs=n_epochs,
        )
        torch.cuda.empty_cache()
        gc.collect()

    _, _, file_suffix = inference(
        model,
        f_prompt_test,
        dataset_id=test_ds,
        lora_version=lora_version,
        output_root=output_root,
        batch_size=bs_test,
        n_test=n_test,
        ds_level=ds_level,
    )

    # clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    gpt_labels = evaluate_gpt4(
        model,
        file_suffix,
        dataset_id=test_ds,
        output_root=output_root,
        n_test=n_test,
        ds_level=ds_level,
        run_id=run_id,
    )
    acc = len([s for s in gpt_labels if extract_label(s[2])]) / len(gpt_labels)

    print(acc)
    print(len(gpt_labels))
    return acc, len(gpt_labels), lora_version


if __name__ == "__main__":

    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    # MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    BS_TRAIN = 12
    BS_TEST = 36
    OUTPUT_ROOT = "output"

    res = []

    # ================= Finetune + Inference + Evaluation =================
    # If want to run inference + evaluation only, set train_ds to None 
    # If want to run inference + evaluation on a lora model, set train_ds to None and lora_version to the desired version, e.g. "2024-08-05_21-50-32"

    ds_level = None
    ds_test = "openai/gsm8k"
    n_test = None
    run_id = 2
    for ds_train in [
        None,  # just inference + evaluation
        "openai/gsm8k"  # finetune on the original train dataset
        "data/gsm8k/lion/2024-08-04_12-15-13/lion_all",  # finetune on the lion dataset, need to replace with the correct path
        "microsoft/orca-math-word-problems-200k",  # finetune on the orca dataset
    ]:
        for n_train in [1000, 10000]:
            acc, total, lora_v = pipeline(
                model=MODEL_ID,
                bs_train=BS_TRAIN,
                train_ds=ds_train,
                n_train=n_train,
                test_ds=ds_test,
                bs_test=BS_TEST,
                n_test=n_test,
                output_root=OUTPUT_ROOT,
                ds_level=ds_level,
                answer_loss_only=True,
                run_id=run_id,
                n_epochs=2,
            )
            print(acc, total)
            print(
                f"Training on {ds_train} ({n_train}) and testing on {ds_test} gave {acc} accuracy."
            )
            res.append((ds_train, n_train, lora_v, acc, total))

    print(res)

