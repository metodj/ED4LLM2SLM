import datetime
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from generation.base import Generator
from generation.orca_math.orca_utils import ama_questions, ama_extract_questions
from utils.gpt4 import gpt4_label_math

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd


class OrcaMathGenerator(Generator):

    def __init__(
        self, dataset: str, N: int, save_dir: str, feedback: bool, r: int, n: int = 1
    ):
        super().__init__(dataset, N, save_dir)
        self.feedback = feedback
        self.r = r  # oversampling ratio
        self.n = n  # number of questions to generate

    def generate(self, slm_preds_path: str = None):
        version = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        dataset = load_dataset(self.dataset, self.data_type)
        dataset = dataset[self.data_split]

        if self.feedback:
            assert slm_preds_path is not None
            assert self.r is not None

            with open(slm_preds_path, "rb") as f:
                gpt_labels = pickle.load(f)

            incorrect_indices = [
                s[0] for s in gpt_labels if "Final Verdict: Correct" not in s[2]
            ]
            correct_indices = [
                s[0] for s in gpt_labels if "Final Verdict: Correct" in s[2]
            ]

            incorrect_questions = dataset.select(incorrect_indices)
            correct_questions = dataset.select(correct_indices)

            questions = []
            for q, n in zip([incorrect_questions, correct_questions], [self.r, 1]):
                _questions = ama_questions(q)
                _questions = [
                    ama_extract_questions(x[1], n_questions=n)
                    for x in _questions
                    if x is not None
                ]
                _questions = [item for sublist in _questions for item in sublist]
                questions.extend(_questions)

        else:
            questions = ama_questions(dataset.select(list(range(self.N))))
            questions = [
                ama_extract_questions(x[1], n_questions=self.n)
                for x in questions
                if x is not None
            ]
            questions = [item for sublist in questions for item in sublist]

        output = gpt4_label_math(questions)
        questions = [x[2] for x in output if x is not None]
        answers = [x[1] for x in output if x is not None]

        assert len(questions) == len(answers)
        if len(questions) != (self.N * self.n):
            print(
                f"Warning: expected {self.N * self.n} questions, got {len(questions)}"
            )

        r = self.r if self.feedback else 0
        ds_name = f"orca_math_{self.N}_{r}_{version}"
        self.save(questions, answers, ds_name, self.save_dir)


if __name__ == "__main__":
    dataset = "openai/gsm8k"
    N = 1000
    save_dir = "data/gsm8k/"

    # no feedback baseline
    generator = OrcaMathGenerator(dataset, N, save_dir, feedback=False, r=None)
    generator.generate()

    # feedback baseline
    generator = OrcaMathGenerator(dataset, N, save_dir, feedback=True, r=3)
    # generator.generate("data/gsm8k/SLM_train_all_gsm8k.pkl")
    generator.generate("output/gpt4-data/orca_math_1000_train.pkl")
