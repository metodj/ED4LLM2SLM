from typing import List
from datasets import Dataset, DatasetDict
import pandas as pd


class Generator:

    def __init__(
        self,
        dataset: str,
        N: int,
        save_dir: str,
        data_type: str = "main",
        data_split: str = "train",
    ):
        self.dataset = dataset
        self.N = N  # size of the seed Dataset
        self.save_dir = save_dir

        self.data_type = data_type
        self.data_split = data_split

    def generate(self):
        raise NotImplementedError

    def save(self, questions: List[str], answers: List[str], name: str, save_dir: str):
        assert len(questions) == len(answers)
        df = pd.DataFrame({"question": questions, "answer": answers})
        dataset = Dataset.from_pandas(df)
        ds = DatasetDict({self.data_split: dataset})
        ds.save_to_disk(f"{save_dir}/{name}")
        print(f"Saved to {save_dir}/{name}")
        print(f"Number of questions: {len(ds[self.data_split])}")
