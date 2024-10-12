import os
os.environ['HF_HOME'] = '/nvmestore/mjazbec/huggingface'

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("Hugging Face token not found in environment variables. Please set HF_AUTH_TOKEN.")
os.environ['HF_TOKEN'] = token
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import pickle
import gc

# Add the directory containing 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
import torch

from inference.inference import inference
from evaluation.gpt4_eval import evaluate_gpt4, QUESTION_DICT, ANSWER_DICT
from utils.prompt_utils import (
    GPT4_ORCA_MATH_SYSTEM_PROMPT,
    orca_math_label_template,
    create_prompt_test_mistral_harness,
)
from utils.gpt4 import gpt4_call, gpt4_label_math, extract_label


def orca_math_label_template(question, answer, generation):
    return f"Question: {question} \n Problem Setter’s answer: {answer} \n Student’s answer: {generation}"


def slm_run(
    model_id: str, dataset: str, save_dir: str, bs: int, data_split: str = "train", N: int = None
):

    assert data_split == "train"

    test_prompt = lambda x: create_prompt_test_mistral_harness(x, question_token=QUESTION_DICT[dataset])

    predictions, _, _ = inference(
        model_id,
        dataset_id=dataset,
        f_prompt=test_prompt,
        batch_size=bs,
        save=False,
        data_split=data_split,
        n_train=N
    )
    gpt_labels = evaluate_gpt4(
        model_id=model_id,
        file_suffix=None,
        answers=predictions,
        dataset_id=dataset,
        data_split=data_split,
        save=False,
        n_train=N
    )
    acc = len([s for s in gpt_labels if "Final Verdict: Correct" in s[2]]) / len(
        gpt_labels
    )

    print(acc)

    if "meta" in model_id:
        _model = "llama"
    elif "mistral" in model_id:
        _model = "mistral"

    with open(f"{save_dir}/SLM_{data_split}_{_model}.pkl", "wb") as f:
        pickle.dump(gpt_labels, f)


def llm_run(ds_name: str, save_dir: str, data_split: str = "train", N: int = None):
    assert data_split == "train"

    dataset = load_dataset(ds_name, "main")
    dataset = dataset[data_split]
    if N:
        dataset = dataset.select(list(range(N)))
    outputs = gpt4_label_math(dataset[QUESTION_DICT[ds_name]])

    # TODO: refactor
    def label_template(output):
        i, answer, _ = output
        return orca_math_label_template(
            dataset[i][QUESTION_DICT[ds_name]], dataset[i][ANSWER_DICT[ds_name]], answer
        )
    gpt4_label_itself = lambda x: gpt4_call(
        x, assistant_message=GPT4_ORCA_MATH_SYSTEM_PROMPT, user_func=label_template
    )

    outputs = gpt4_label_itself(outputs)

    # TODO: refactor such that this will not be needed...
    outputs = [(x[2][0], x[2][1], x[1]) for x in outputs if x is not None]

    acc = len([s for s in outputs if "Final Verdict: Correct" in s[2]]) / len(
        outputs
    )
    acc2 = len([s for s in outputs if extract_label(s[2])]) / len(outputs)

    print(acc, acc2)

    with open(f"{save_dir}/LLM_{data_split}_2.pkl", "wb") as f:
        pickle.dump(outputs, f)

if __name__ == "__main__":

    # # ==================== compute SLM/LLM predictions ===================

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    for dataset in ["openai/gsm8k", "hendrycks/competition_math"]:
        _dataset = dataset.split("/")[-1]
        save_dir = f"data/{_dataset}"
        
        bs = 50
        N = None

        slm_run(model_id, dataset, save_dir, bs, N=N)
        torch.cuda.empty_cache()
        gc.collect()
        llm_run(dataset, save_dir, N=N)


    # # ==================== read in SLM/LLM predictions ===================
    # # read in "data/competition_math/LLM_train.pkl"

    # with open("data/gsm8k/LLM_train_mistralai.pkl", "rb") as f:
    # # with open("data/gsm8k/SLM_train.pkl", "rb") as f:
    # # with open("data/gsm8k/SLM_train_llama.pkl", "rb") as f:
    # # with open("data/competition_math/LLM_train.pkl", "rb") as f:
    #     output = pickle.load(f)
    
    # output = [x for x in output if x is not None]
    # print(len([s for s in output if extract_label(s[2])]) / len(output))
    # print(len(output))


    

