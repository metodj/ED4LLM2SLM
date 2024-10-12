"""
Code adapted from: https://github.com/YJiangcm/Lion
"""


import datetime
import os
import sys
import pickle
import random
import time
import json
from functools import partial
from multiprocessing import Pool
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from generation.base import Generator
from generation.lion.lion_utils import (
    read_qa,
    parse_score,
    referee,
    get_json_list,
    get_json_list2,
    jdump,
    jload,
    lion_generate,
    encode_prompt,
    post_process_gpt3_response,
)
from utils.gpt4 import gpt4_label_math

from datasets import DatasetDict, load_from_disk, concatenate_datasets, load_dataset
import pandas as pd
import numpy as np
import tqdm
from rouge_score import rouge_scorer


class LionGenerator(Generator):

    def __init__(
        self,
        dataset: str,
        N: int,  # number of new synthetic samples
        N_seed: int,  # number of seed examples
        save_dir: str,
        level: int = None,
    ):
        super().__init__(dataset, N, save_dir)

        self.N_seed = N_seed

        assert dataset in ["openai/gsm8k", "hendrycks/competition_math"]
        if dataset == "openai/gsm8k":
            self.level = None
            self.ques_token = "question"
        elif dataset == "hendrycks/competition_math":
            self.level = level
            self.ques_token = "problem"

    def referee(
        self,
        version: str,
        ques: List[str],
        ans1: List[str],
        ans2: List[str],
        order: str,
    ):
        output_review_file_path = (
            f"{self.save_dir}/{version}/referee_{order}.txt"
        )

        # check if # of questions, answers are the same
        assert len(ans1) == len(ans2) == len(ques)
        question_idx_list = list(range(len(ques)))

        data = list(zip(ques, ans1, ans2))
        reviews = referee(data)
        reviews = sorted(reviews, key=lambda x: x[0])
        
        reviews_ids = [x[0] for x in reviews]
        ques = [ques[idx] for idx in reviews_ids]
        ans1 = [ans1[idx] for idx in reviews_ids]
        ans2 = [ans2[idx] for idx in reviews_ids]

        reviews = [x[1] for x in reviews]
        assert len(ques) == len(reviews)

        with open(f"{output_review_file_path}", "w") as output_review_file:
            for idx, review in enumerate(reviews):
                scores = parse_score(review)
                output_review_file.write(
                    json.dumps({"score": scores, "text": review}) + "\n"
                )

        # else:
        ### recheck
        old_question_idx_list = question_idx_list

        reviews = get_json_list2(output_review_file_path)
        new_question_idx_list = []
        for idx in range(len(reviews)):
            if parse_score(reviews[idx]["text"]) == [-1, -1]:
                new_question_idx_list.append(idx)

        
        n_tries = 0
        while len(new_question_idx_list) < len(old_question_idx_list):
            n_tries += 1
            if n_tries < 3:
                fail_score = -1
            else:
                fail_score = 0

            new_ques, new_ans1, new_ans2 = (
                [ques[idx] for idx in new_question_idx_list],
                [ans1[idx] for idx in new_question_idx_list],
                [ans2[idx] for idx in new_question_idx_list],
            )
            new_data = list(zip(new_ques, new_ans1, new_ans2))
            new_reviews = referee(new_data)
            new_reviews = sorted(new_reviews, key=lambda x: x[0])
            new_reviews = [x[1] for x in new_reviews]

            assert len(new_question_idx_list) == len(new_reviews)
            for idx, review in enumerate(new_reviews):
                scores = parse_score(review, fail_score=fail_score)
                reviews[new_question_idx_list[idx]]["score"] = scores
                reviews[new_question_idx_list[idx]]["text"] = review

            with open(f"{output_review_file_path}", "w") as output_review_file:
                for idx, review in enumerate(reviews):
                    output_review_file.write(
                        json.dumps({"score": review["score"], "text": review["text"]})
                        + "\n"
                    )

            old_question_idx_list = new_question_idx_list

            new_question_idx_list = []
            for idx in range(len(reviews)):
                if parse_score(reviews[idx]["text"], fail_score=fail_score) == [-1, -1]:
                    new_question_idx_list.append(idx)

    def discriminate(
        self, version: str, ques: List[str], ans1: List[str], ans2: List[str]
    ):
        review12 = get_json_list2(f"{self.save_dir}/{version}/referee_12.txt")
        review21 = get_json_list2(f"{self.save_dir}/{version}/referee_21.txt")

        # ans1: LLM
        # ans2: SLM

        ans1_score12 = [i["score"][1] for i in review12]
        ans2_score12 = [i["score"][0] for i in review12]
        review_text12 = [i["text"] for i in review12]

        print("LLM: ", np.mean(ans1_score12))
        print("SLM: ", np.mean(ans2_score12))

        ans1_score21 = [i["score"][0] for i in review21]
        ans2_score21 = [i["score"][1] for i in review21]
        review_text21 = [i["text"] for i in review21]

        print("LLM: ", np.mean(ans1_score21))
        print("SLM: ", np.mean(ans2_score21))

        ans1_score = [
            (i["score"][1] + j["score"][0]) / 2 for i, j in zip(review12, review21)
        ]
        ans2_score = [
            (i["score"][0] + j["score"][1]) / 2 for i, j in zip(review12, review21)
        ]

        review_score_diff = [(i - j) for i, j in zip(ans1_score, ans2_score)]

        referee = pd.DataFrame(
            {
                "instruction": ques,
                "assist1": ans2,  # Input to the function is ans1 - SLM and ans2 - LLM. However, the scores above are calculated as ans1 - LLM and ans2 - SLM. Hence, we do "assist1": ans2 and "assist2": ans1
                "assist2": ans1,
                "assist1_score12": ans1_score12,
                "assist2_score12": ans2_score12,
                "review_text12": review_text12,
                "assist1_score21": ans1_score21,
                "assist2_score21": ans2_score21,
                "review_text21": review_text21,
                "assist1_score": ans1_score,
                "assist2_score": ans2_score,
                "review_score_diff": review_score_diff,
            }
        )

        referee = referee.sort_values(
            by=["review_score_diff", "assist1_score"], ascending=False
        )
        referee = referee.reset_index(drop=False)

        hard_instructions = referee[(referee["review_score_diff"] >= 1)]
        easy_instructions = referee[(referee["review_score_diff"] < 1)]

        print(
            f"Number of hard instructions: {len(hard_instructions)}, Number of easy instructions: {len(easy_instructions)}"
        )

        # save the identified hard instructions
        hard_instructions = hard_instructions.reset_index(drop=False)

        save_path = f"{self.save_dir}/{version}/discrimination"
        hard_save_path = f"{save_path}_hard.txt"
        easy_save_path = f"{save_path}_easy.txt"

        with open(hard_save_path, "w") as output_hard_file:
            for i in range(len(hard_instructions)):
                output_hard_file.write(
                    json.dumps(
                        {
                            "instruction": hard_instructions.iloc[i]["instruction"],
                            "assist1": hard_instructions.iloc[i]["assist1"],
                            "assist2": hard_instructions.iloc[i]["assist2"],
                            "assist1_score": hard_instructions.iloc[i]["assist1_score"],
                            "assist2_score": hard_instructions.iloc[i]["assist2_score"],
                            "review_score_diff": hard_instructions.iloc[i][
                                "review_score_diff"
                            ],
                        }
                    )
                    + "\n"
                )

        # save the identified easy instructions
        easy_instructions = easy_instructions.reset_index(drop=False)
        with open(easy_save_path, "w") as output_easy_file:
            for i in range(len(easy_instructions)):
                output_easy_file.write(
                    json.dumps(
                        {
                            "instruction": easy_instructions.iloc[i]["instruction"],
                            "assist1": easy_instructions.iloc[i]["assist1"],
                            "assist2": easy_instructions.iloc[i]["assist2"],
                            "assist1_score": easy_instructions.iloc[i]["assist1_score"],
                            "assist2_score": easy_instructions.iloc[i]["assist2_score"],
                            "review_score_diff": easy_instructions.iloc[i][
                                "review_score_diff"
                            ],
                        }
                    )
                    + "\n"
                )

    def create_questions(
        self,
        version: str,
        all_instruction_data: List[str],
        prompt_type: str,
        req_bs: int = 100,
        n_prompt_inst: int = 1,
        num_cpus: int = 8,
    ):
        output_dir = f"{self.save_dir}/{version}"
        seed_path = f"{output_dir}/discrimination_{prompt_type}.txt"
        seed_tasks = [json.loads(l) for l in open(seed_path, "r")]
        seed_instruction_data = [
            {"instruction": t["instruction"], "output": t["assist1"]}
            for t in seed_tasks
        ]
        print(f"Loaded {len(seed_instruction_data)} seed instructions")

        print(f"Loaded {len(all_instruction_data)} all instructions")

        os.makedirs(output_dir, exist_ok=True)
        request_idx = 0
        # load the LM-generated instructions
        machine_instruction_data = []
        if os.path.exists(os.path.join(output_dir, f"questions_{prompt_type}.json")):
            machine_instruction_data = jload(
                os.path.join(output_dir, f"questions_{prompt_type}.json")
            )
            print(
                f"Loaded {len(machine_instruction_data)} machine-generated instructions"
            )

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        # now let's generate new instructions!
        progress_bar = tqdm.tqdm(total=self.N)
        if machine_instruction_data:
            progress_bar.update(len(machine_instruction_data))

        all_instructions = [d for d in all_instruction_data] + [
            d["instruction"] for d in machine_instruction_data
        ]
        all_instruction_tokens = [
            scorer._tokenizer.tokenize(inst) for inst in all_instructions
        ]

        while len(machine_instruction_data) < self.N:
            request_idx += 1

            batch_inputs = []
            inst_i_os = []
            for _ in range(req_bs):
                # only sampling from the seed tasks
                prompt_instructions = random.sample(
                    seed_instruction_data, n_prompt_inst
                )
                prompt, inst_i_o = encode_prompt(
                    prompt_instructions, prompt_type=prompt_type
                )
                batch_inputs.append(prompt)
                inst_i_os.extend(inst_i_o)

            request_start = time.time()

            results = lion_generate(batch_inputs)
            results = sorted(results, key=lambda x: x[0])

            request_duration = time.time() - request_start

            process_start = time.time()
            instruction_data = []

            assert len(results) == len(inst_i_os)
            for idx, result in enumerate(results):
                new_instructions = post_process_gpt3_response(result, inst_i_os[idx])
                instruction_data += new_instructions

            total = len(instruction_data)
            keep = 0
            for instruction_data_entry in instruction_data:
                # computing similarity with the pre-tokenzied instructions
                new_instruction_tokens = scorer._tokenizer.tokenize(
                    instruction_data_entry["instruction"]
                )
                with Pool(num_cpus) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, new_instruction_tokens),
                        all_instruction_tokens,
                    )
                rouge_scores = [score.fmeasure for score in rouge_scores]
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                if max(rouge_scores) > 0.7:
                    continue
                else:
                    keep += 1

                instruction_data_entry["most_similar_instructions"] = (
                    most_similar_instructions
                )
                instruction_data_entry["avg_similarity_score"] = float(
                    np.mean(rouge_scores)
                )
                machine_instruction_data.append(instruction_data_entry)
                all_instructions.append(instruction_data_entry["instruction"])
                all_instruction_tokens.append(new_instruction_tokens)
                progress_bar.update(1)
            process_duration = time.time() - process_start
            print(
                f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s"
            )
            print(f"Generated {total} instructions, kept {keep} instructions")
            jdump(
                machine_instruction_data,
                os.path.join(output_dir, f"questions_{prompt_type}.json"),
            )

    def label(self, version: str, merge: bool = True):
        save_dir = f"{self.save_dir}/{version}"
        for q_type in ["easy", "hard"]:
            question_jsons = get_json_list(f"{save_dir}/questions_{q_type}.json")

            ques = [i["instruction"] for i in question_jsons]
            output = gpt4_label_math(ques)
            questions = [x[2] for x in output if x is not None]
            answers = [x[1] for x in output if x is not None]

            self.save(questions, answers, f"lion_{q_type}", save_dir)

        if merge:
            # merge datasets
            ds_easy = load_from_disk(f"{save_dir}/lion_easy")
            ds_hard = load_from_disk(f"{save_dir}/lion_hard")

            ds = concatenate_datasets(
                [ds_easy[self.data_split], ds_hard[self.data_split]]
            )
            ds = DatasetDict({self.data_split: ds})
            ds.save_to_disk(f"{save_dir}/lion_all")
            print(f"Merged datasets saved to {save_dir}/lion_all")
            print(f"Number of questions: {len(ds[self.data_split])}")

    def generate(self, slm_preds_path: str, llm_preds_path: str):
        version = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.level is not None:
            version += f"_level{self.level}"
        os.makedirs(f"{self.save_dir}/{version}", exist_ok=True)

        ques, ans_slm, ans_llm = read_qa(
            self.dataset,
            slm_preds_path,
            llm_preds_path,
            N=self.N_seed,
            level=self.level,
            ques_token=self.ques_token,
        )

        print(len(ques), len(ans_slm), len(ans_llm))
        print(ques[:1], ans_slm[:1], ans_llm[:1])

        self.referee(version, ques, ans_slm, ans_llm, "12")
        self.referee(version, ques, ans_llm, ans_slm, "21")

        self.discriminate(version, ques, ans_slm, ans_llm)

        self.create_questions(version, ques, prompt_type="hard")
        self.create_questions(version, ques, prompt_type="easy")

        self.label(version)


if __name__ == "__main__":
    # ====== MISTRAL GSM8k =======
    dataset = "openai/gsm8k"
    N_seed = 7500
    N = 10000
    save_dir = "data/gsm8k/lion"

    generator = LionGenerator(dataset=dataset, N=N, N_seed=N_seed, save_dir=save_dir)
    generator.generate(
        slm_preds_path="data/gsm8k/SLM_train.pkl",
        llm_preds_path="data/gsm8k/LLM_train.pkl",
    )
    

    # ====== Mistral7B MATH =======
    # dataset = "hendrycks/competition_math"
    # N = 5000
    # N_seed = 2500
    # save_dir = "data/competition_math/lion"

    # # for level in [1, 3, 5]:
    # for level in [5, 3, 1]:
    #     generator = LionGenerator(dataset=dataset, N=N, N_seed=N_seed, save_dir=save_dir, level=level)
    #     generator.generate(
    #         slm_preds_path="data/competition_math/SLM_train.pkl",
    #         llm_preds_path="data/competition_math/LLM_train.pkl",
    #     )

    # # ===== Llama3 GSM8k =========
    # dataset = "openai/gsm8k"
    # N_seed = 7500
    # N = 10000
    # save_dir = "data/gsm8k/lion/llama"

    # generator = LionGenerator(dataset=dataset, N=N, N_seed=N_seed, save_dir=save_dir)
    # generator.generate(
    #     slm_preds_path="data/gsm8k/SLM_train_llama.pkl",
    #     llm_preds_path="data/gsm8k/LLM_train.pkl",
    # )

    


