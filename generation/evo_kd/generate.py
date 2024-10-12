import datetime
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from generation.base import Generator
from generation.evo_kd.evokd_utils import evo_kd_extract_questions, evo_kd_questions
from utils.gpt4 import gpt4_label_math

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    load_from_disk,
    concatenate_datasets,
)
import pandas as pd
import numpy as np


class EvoKDGenerator(Generator):

    def __init__(self, dataset: str, N: int, save_dir: str, n: int = 2):
        super().__init__(dataset, N, save_dir)
        self.n = n  # number of questions to generate

        # TODO: modify evo_kd_prompt, evo_kd_questions to handle n > 2
        assert self.n == 2

    def generate(
        self,
        slm_preds_path: str,
        seed: int = 42,
        merge: bool = True,
        n_examples: int = 2,
    ):
        version = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        np.random.seed(seed)

        dataset = load_dataset(self.dataset, self.data_type)
        dataset = dataset[self.data_split]

        with open(slm_preds_path, "rb") as f:
            gpt_labels = pickle.load(f)
        id_map = {i[0]: j for j, i in enumerate(gpt_labels)}

        incorrect_indices = [
            s[0] for s in gpt_labels if "Final Verdict: Correct" not in s[2]
        ]
        correct_indices = [s[0] for s in gpt_labels if "Final Verdict: Correct" in s[2]]

        incorrect_questions = dataset.select(incorrect_indices)
        correct_questions = dataset.select(correct_indices)

        inputs = []
        for _ in range(int(self.N / self.n) + 10):
            corr_ids = np.random.choice(len(correct_questions), self.n, replace=False)
            incorr_ids = np.random.choice(
                len(incorrect_questions), self.n, replace=False
            )
            corr_ids = [int(x) for x in corr_ids]
            incorr_ids = [int(x) for x in incorr_ids]

            inputs.append(
                {
                    "correct_questions": [
                        correct_questions[corr_ids[i]]["question"]
                        for i in range(n_examples)
                    ],
                    "correct_answers": [
                        gpt_labels[id_map[correct_indices[corr_ids[i]]]][1]
                        for i in range(n_examples)
                    ],
                    "wrong_questions": [
                        incorrect_questions[incorr_ids[i]]["question"]
                        for i in range(n_examples)
                    ],
                    "wrong_answers": [
                        gpt_labels[id_map[incorrect_indices[incorr_ids[i]]]][1]
                        for i in range(n_examples)
                    ],
                }
            )

        _questions = evo_kd_questions(inputs)
        _questions = [
            evo_kd_extract_questions(x[1]) for x in _questions if x is not None
        ]
        for q_type in ["hard", "easy"]:
            _q = [x[q_type] for x in _questions]
            _q = [item for sublist in _q for item in sublist]

            outputs = gpt4_label_math(_q)

            _q = [x[2] for x in outputs if x is not None]
            _a = [x[1] for x in outputs if x is not None]

            name = f"evokd_{q_type}_{version}"
            self.save(_q, _a, name, self.save_dir)

        if merge:
            dataset_easy = load_from_disk(f"{self.save_dir}/evokd_easy_{version}")
            dataset_hard = load_from_disk(f"{self.save_dir}/evokd_hard_{version}")
            dataset = concatenate_datasets(
                [dataset_easy[self.data_split], dataset_hard[self.data_split]]
            )
            ds = DatasetDict({self.data_split: dataset})
            ds.save_to_disk(f"{self.save_dir}/evokd_{version}")
            print(f"Merged datasets saved to {self.save_dir}/evokd_{version}")
            print(f"Number of questions: {len(ds[self.data_split])}")


if __name__ == "__main__":
    dataset = "openai/gsm8k"
    N = 1000
    save_dir = "data/gsm8k/"

    # no feedback baseline
    generator = EvoKDGenerator(dataset, N, save_dir)
    # generator.generate("data/gsm8k/SLM_train_all_gsm8k.pkl")
    generator.generate("output/gpt4-data/orca_math_1000_train.pkl")