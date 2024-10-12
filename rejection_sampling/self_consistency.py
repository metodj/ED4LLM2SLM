import os
import sys
import json
import re

import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict, load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation.lion.lion_utils import get_json_list
from utils.gpt4 import gpt4_label_math, gpt4_call, extract_label
from utils.prompt_utils import GPT4_ORCA_MATH_SYSTEM_PROMPT


def labels_consistency(
    save_dir: str,
    data_type: str,
    M: int = 3,
    N: int = None,
    q_token: str = "question",
    gt_ques: bool = False,
    level: int = None,
):
    if gt_ques:
        ques = load_dataset(data_type)["train"]
        if level:
            ques = ques.filter(lambda x: x["level"] == f"Level {level}")
        _save_dir = save_dir
    else:

        ques = load_from_disk(f"{save_dir}/{data_type}")["train"]
        _save_dir = f"{save_dir}/consistency_labels/{data_type}"

    if N:
        ques = ques.select(list(range(N)))

    ques = ques[q_token]

    for i in range(M):
        labels = gpt4_label_math(ques)
        labels = sorted(labels, key=lambda x: x[0])

        # save labels

        with open(f"{_save_dir}/labels_{i+1}.json", "w") as f:
            for l in labels:
                f.write(json.dumps(l) + "\n")


def labels_merge(save_dir: str, gt_ds: str, M: int = 3, level: int = None, a_token: str = "solution"):
    df = pd.DataFrame(
        columns=[f"question{i+1}" for i in range(M)]
        + [f"answer{i+1}" for i in range(M)]
    )

    for i in range(M):

        # Read the file line by line
        with open(f"{save_dir}/labels_{i + 1}.json", "r") as file:
            for j, line in enumerate(file):
                try:
                    data = json.loads(line)
                    df.loc[j, f"question{i+1}"] = data[2]
                    df.loc[j, f"answer{i+1}"] = data[1]

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

    # check that all questions are the same
    for i in range(1, M):
        assert (df[f"question1"] == df[f"question{i+1}"]).all()

    # drop columns question2, question3, ...
    df = df.drop(columns=[f"question{i+1}" for i in range(1, M)])
    df = df.rename(columns={"question1": "question"})

    # add a column with GT answer
    if gt_ds:
        gt_ds = load_dataset(gt_ds)["train"]
        if level:
            gt_ds = gt_ds.filter(lambda x: x["level"] == f"Level {level}")
        indices = df.index.tolist()
        gt_ds = gt_ds.select(indices)
        df["answer_gt"] = gt_ds[a_token]

    df.to_csv(f"{save_dir}/labels_merged.csv", index=False)
    # df.to_csv(f"{save_dir}/labels_merged_5.csv", index=False)

    return df


CONSISTENCY_USER_PROMPT = "Please verify if all the answers reach the same final answer. If the answer indeed agree return 1, else return 0. Please first output a single line containing the value indicating the agreement (0 or 1). In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."


def consistency_user_prompt(row):
    return CONSISTENCY_USER_PROMPT


def consistency_system_prompt(row, q_token="question", a_token="answer", M: int = 3):
    q = row[q_token]
    a = []
    prompt = """
        We would like to request your feedback on the agreement between the AI assistant's answers in response to the mathematical question displayed following.\n
    """
    prompt += f"Question: {q}\n"
    for m in range(M):
        a = row[f"{a_token}{m + 1}"]
        prompt += f"Answer {m+1}: {a}\n"

    return prompt


consistency_filter = lambda x: gpt4_call(
    x, system_func=consistency_system_prompt, user_func=consistency_user_prompt
)

def orca_math_label_template(question, answer, generation):
    return f"Question: {question} \n Problem Setter’s answer: {answer} \n Student’s answer: {generation}"


def label_template(output):
    question, preds, answer = output
    return orca_math_label_template(question, preds, answer)


gpt4_label_itself = lambda x: gpt4_call(
    x, assistant_message=GPT4_ORCA_MATH_SYSTEM_PROMPT, user_func=label_template
)

def label_all(df_path: str):

    df = pd.read_csv(df_path + '/labels_merged.csv')

    ans = [x for x in df.columns if "answer" in x and "gt" not in x]

    for i in range(len(ans)):
        outputs = gpt4_label_itself(df[["question", "answer_gt", ans[i]]].values.tolist())

        outputs = sorted(outputs, key=lambda x: x[0])
        outputs = [(x[0], x[1], extract_label(x[1])) for x in outputs if x is not None]

        # save as a csv
        outputs = pd.DataFrame(outputs, columns=["id", "label", "score"])
        outputs.to_csv(f"{df_path}/eval_{i+1}.csv", index=False)

        print(outputs)

    return df

if __name__ == "__main__":

    # # ======= self-consistency (synthetic questions) ======

    # SAVE_DIRS = [
    #     "data/competition_math/lion/2024-07-10_17-50-12_level1",
    #     "data/competition_math/lion/2024-07-10_19-33-49_level3",
    #     "data/competition_math/lion/2024-07-10_22-00-11_level5",
    #     "data/gsm8k/lion/2024-07-06_18-20-59",
    # ]
    # LION_TYPE = "lion_all"

    # for save_dir in SAVE_DIRS:
    #     print(f"Processing {save_dir}")
    #     _save_dir = f"{save_dir}/consistency_labels/{LION_TYPE}"
    #     labels_consistency(save_dir, LION_TYPE)
    #     labels_merge(_save_dir)

    #     # df = pd.read_csv(f"{_save_dir}/labels_merged.csv")
    #     df = pd.read_csv(f"{_save_dir}/labels_merged_5.csv")

    #     dataset = Dataset.from_pandas(df)
    #     df = consistency_filter(dataset)

    #     df = sorted(df, key=lambda x: x[0])
    #     df = [(x[0], int(re.search(r"\d+", x[1]).group())) for x in df]
    #     df = pd.DataFrame(df, columns=["id", "score"])
    #     assert len(df) == len(dataset)
    #     # df.to_csv(f"{_save_dir}/consistency_scores.csv", index=False)
    #     df.to_csv(f"{_save_dir}/consistency_scores_5.csv", index=False)

    #     print(sum(df["score"] == 1) / len(df))

    #     # read in the dataset
    #     dataset = load_from_disk(f"{save_dir}/{LION_TYPE}")

    #     # filter the dataset where consistency score is 1
    #     ids = df[df["score"] == 1]["id"].tolist()
    #     dataset = dataset["train"].select(ids)

    #     # save the filtered dataset
    #     ds = DatasetDict({"train": dataset})
    #     # ds.save_to_disk(f"{save_dir}/{LION_TYPE}_consistent")
    #     ds.save_to_disk(f"{save_dir}/{LION_TYPE}_consistent_5")

    # ======= self-consistency (real questions) ======

    save_dir = "data/self-consistency"

    for ds, level in zip(["hendrycks/competition_math", "hendrycks/competition_math"], [1, 3]):
        print(f"Processing {ds} level {level}")
        _ds = ds.split("/")[1]
        _save_dir = f"{save_dir}/{_ds}"
        if level:
            _save_dir += f"/{level}"
        labels_consistency(_save_dir, ds, gt_ques=True, q_token="problem", level=level)
        labels_merge(_save_dir, gt_ds=ds, level=level)

        df = pd.read_csv(f"{_save_dir}/labels_merged.csv")

        dataset = Dataset.from_pandas(df)
        df = consistency_filter(dataset)

        df = sorted(df, key=lambda x: x[0])
        df = [(x[0], int(re.search(r"\d+", x[1]).group()), x[1]) for x in df]
        df = pd.DataFrame(df, columns=["id", "score", "output"])
        assert len(df) == len(dataset)
        df.to_csv(f"{_save_dir}/consistency_scores.csv", index=False)

        print(sum(df["score"] == 1) / len(df))

        # label
        label_all(_save_dir)
