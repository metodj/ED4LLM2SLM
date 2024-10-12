import os
import sys

# Add the directory containing 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import re
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from utils.gpt4 import gpt4_call, gpt4_label_math


ALPAGASUS_USER_PROMPT = "Please rate according to the correctness of the answer for the given question. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of the correctness. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."

def alpagasus_user_prompt(row):
    return ALPAGASUS_USER_PROMPT

VERIFY_USER_PROMPT = "Please check if the provided answer is correct for the given question. If the answer indeed solves the question give it a score of 1, else give it a score of 0. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."

def verify_user_prompt(row):
    return VERIFY_USER_PROMPT

def alpagasus_system_prompt(row, q_token="question", a_token="answer"):
    q, a = row[q_token], row[a_token]
    prompt = """
        We would like to request your feedback on the quality of AI assistant's answer in response to the mathematical question displayed following.\n
    """
    prompt += f"Question: {q}\n"
    prompt += f"Answer: {a}\n"

    return prompt

def ds_filter(
    ds_path: str, filter_output_path: str, threshold: int = 4.5, ds_split: str = "train"
):

    # load ds
    dataset = load_from_disk(ds_path)
    print(f"Length of dataset: {len(dataset[ds_split])}")

    # load filter_output
    with open(filter_output_path, "rb") as f:
        filter_output = pickle.load(f)

    # filter the dataset
    scores = [(x[0], re.search(r"\d+", x[1]).group()) for x in filter_output]
    ids = [x[0] for x in scores if int(x[1]) >= threshold]
    dataset = dataset[ds_split].select(sorted(ids))
    print(f"Length of filtered dataset: {len(dataset)}")

    # save the filtered dataset
    ds = DatasetDict({ds_split: dataset})
    ds.save_to_disk(ds_path + "_filtered")
    print(f"Filtered dataset saved to {ds_path}_filtered")


alpagasus_filter = lambda x: gpt4_call(
    x, system_func=alpagasus_system_prompt, user_func=alpagasus_user_prompt
)

if __name__ == "__main__":

    # ======= FILTER SYNTHETIC DATA =================

    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset["train"]["question"]

    for data_id in [
        "2024-07-10_22-00-11_level5",
        "2024-07-10_19-33-49_level3",
        "2024-07-10_17-50-12_level1",
    ]:

        dataset = load_from_disk(f"data/competition_math/lion/{data_id}/lion_hard")
        dataset = dataset["train"]
        # print(dataset[1])
        res = alpagasus_filter(dataset)

        # data_id = data_id.replace("/", "_")
        with open(f"data/filtering/{data_id}_lion_hard.pkl", "wb") as f:
            pickle.dump(res, f)

        ds_filter(
        f"data/competition_math/lion/{data_id}/lion_hard",
        f"data/filtering/{data_id}_lion_hard.pkl")

    # # ======= FILTER REAL DATA (SANITY CHECK) =================

    # N = 500
    # SEED = 42
    # SAVE_ROOT = "data/filtering/debug"

    # DATA = "gsm8k"
    # # DATA = "competition_math"

    # # for ds_test in ["GT_correct", "SLM_wrong", "LLM_wrong"]:
    # # for ds_test in ["SLM_wrong", "LLM_wrong"]:
    # for ds_test in ["SLM_correct", "LLM_correct"]:

    #     ds_test = DATA + "_" + ds_test

    #     for filter_name, user_prompt in zip(["alpagasus", "verify"], [alpagasus_user_prompt, verify_user_prompt]):

    #         # dataset = load_dataset(ds_test, "main")
    #         dataset = load_from_disk(f"{SAVE_ROOT}/valid_data/{ds_test}")
    #         dataset = dataset["train"]

    #         if N < len(dataset):
    #             dataset = dataset.shuffle(seed=SEED).select(list(range(N)))

    #         _filter = lambda x: gpt4_call(
    #             x, system_func=alpagasus_system_prompt, user_func=user_prompt
    #         )

    #         res = _filter(dataset)
    #         with open(f"{SAVE_ROOT}/filter_res/{filter_name}/{ds_test}.pkl", "wb") as f:
    #             pickle.dump(res, f)

    #         scores = [(x[0], re.search(r"\d+", x[1]).group()) for x in res]
    #         print(ds_test)
    #         if filter_name == "alpagasus":
    #             print(len([x[0] for x in scores if int(x[1]) >= 4.5]), len(scores))
    #         else:
    #             print(len([x[0] for x in scores if int(x[1]) == 1]), len(scores))




   
    
