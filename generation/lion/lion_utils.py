"""
Code adapted from: https://github.com/YJiangcm/Lion
"""

import random
import os
import json
import pickle
import io
import sys
import string
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.gpt4 import gpt4_call


def get_json_list2(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


def read_qa(
    dataset: str,
    answers1: str,
    answers2: str,
    N: int = 1000,
    seed: int = 42,
    answer_id: int = 1,
    data_split: str = "train",
    data_type: str = "main",
    ques_token: str = "question",
    level: int = None
):
    random.seed(seed)
    dataset = load_dataset(dataset, data_type)
    dataset = dataset[data_split]
    if level is not None:
        level_ids = [i for i, x in enumerate(dataset) if x["level"] == f"Level {level}"]
        # dataset = dataset.filter(lambda x: x["level"] == f"Level {level}")

    with open(answers1, "rb") as f:
        output1 = pickle.load(f)

    with open(answers2, "rb") as f:
        output2 = pickle.load(f)

    ids1 = [x[0] for x in output1]
    ids2 = [x[0] for x in output2]
    ids = list(set(ids1).intersection(ids2))
    if level is not None:
        # only keep the ids that are in the level
        ids = [i for i in ids if i in level_ids]
    if N < len(ids):
        ids = random.sample(ids, N)
    ids = sorted(ids)

    output1 = [x for x in output1 if x[0] in ids]
    output1 = sorted(output1, key=lambda x: x[0])

    output2 = [x for x in output2 if x[0] in ids]
    output2 = sorted(output2, key=lambda x: x[0])

    output1 = [x[answer_id] for x in output1]
    output2 = [x[answer_id] for x in output2]
    dataset = dataset.select(ids)
    dataset = dataset[ques_token]

    print("Original dataset size: ", len(dataset))

    return dataset, output1, output2


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_json_list(file_path):
    with open(file_path, "r") as fcc_file:
        json_list = json.load(fcc_file)
    return json_list


def parse_score(review, fail_score=-1):
    try:
        score1 = review.split("\n")[-2]
        score2 = review.split("\n")[-1]

        if "Assistant 1" in score1.split(":")[0]:
            score1 = score1.split(":")[-1].strip()
        else:
            print(f"Failed to parse scores from {review}")
            return [fail_score, fail_score]

        if "Assistant 2" in score2.split(":")[0]:
            score2 = score2.split(":")[-1].strip()
        else:
            print(f"Failed to parse scores from {review}")
            return [fail_score, fail_score]

        return [float(score1), float(score2)]

    except:
        print(f"Failed to parse scores from {review}")
        return [fail_score, fail_score]


def gen_prompt(inputs):
    ques, ans1, ans2 = inputs

    prompt_template = "[Instruction]\n{instruction}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n"

    default_prompt = """We would like to request your feedback on the performance of two AI assistants in response to the user instruction displayed above.
    
    Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

    Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Then, output two lines indicating the scores for Assistant 1 and 2, respectively.

    Output with the following format:
    Evaluation evidence: <your evaluation explanation here>
    Score of the Assistant 1: <score>
    Score of the Assistant 2: <score>"""

    prompt = prompt_template.format(
        instruction=ques, answer_1=ans1, answer_2=ans2, prompt=default_prompt
    )

    return prompt

referee = lambda x: gpt4_call(x, assistant_message="You are a helpful and precise assistant for checking the quality of the answer.", user_func=gen_prompt)


lion_generate = lambda x: gpt4_call(x, assistant_message=None, user_func=None, return_str=False)

def encode_prompt(prompt_instructions, prompt_type):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(f"generation/lion/prompts/prompt_{prompt_type}.txt").read() + "\n"
    inst_i_o = []
    for _, task_dict in enumerate(prompt_instructions):
        inst_i_o.append({"instruction": task_dict["instruction"], "output": task_dict["output"]})
        instruction = task_dict["instruction"]
        prompt += f"{instruction}\n"
    prompt += "\n#Created Instruction#:"
    return prompt, inst_i_o


def post_process_gpt3_response(response, inst_i_o):
    if response is None:
        return []
    if response[1] is None:
        return []
    
    _, response, _ = response
    try:
        instruction = response["message"]['content']
    except:
        return []

    if 'instruction' in instruction.lower():
        return []
    # if the decoding stops due to length, the last example is likely truncated so we discard it
    if response["finish_reason"] == "length":
        return []
    # filter out too short instructions
    if len(instruction.split()) <= 3:
        return []
    # filter those starting with punctuation
    if instruction[0] in string.punctuation:
        return []
    # filter those starting with non-english character
    if not instruction[0].isascii():
        return []
        
    instructions = [{
                    "seed_instruction": inst_i_o["instruction"],
                    "seed_output": inst_i_o["output"],
                    "instruction": instruction,
                    }]
    return instructions