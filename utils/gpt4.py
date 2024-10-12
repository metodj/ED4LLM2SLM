from openai import AzureOpenAI
import concurrent.futures
from openai import OpenAIError

import os
from dotenv import load_dotenv
import time
from typing import List, Callable

load_dotenv()  # take environment variables from .env
KEY = os.getenv("OPENAI_API_KEY")
ENDPOINT = os.getenv("OPENAI_ENDPOINT")

assert KEY
assert ENDPOINT

import sys

# Add the directory containing 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.prompt_utils import GPT4_MATH_LABEL_SYSTEM_PROMPT


def extract_label(x):
    candidates = [
        "final verdict:\ncorrect",
        "final verdict: correct",
        "final verdict:correct",
        "final verdict:\n\ncorrect",
        "final verdict: \ncorrect",
    ]
    for c in candidates:
        if c in x.lower():
            return 1
    return 0


def gpt4_call(
    dataset: List,
    assistant_message: str = None,
    user_func: Callable = None,
    system_func: Callable = None,
    return_str: bool = True,
    api_key: str = KEY,
    endpoint: str = ENDPOINT,
):

    client = AzureOpenAI(
        api_key=api_key, api_version="2024-02-01", azure_endpoint=endpoint
    )
    deployment_name = "gpt-4-turbo"

    MAX_RETRIES = 100  # Maximum number of retries
    INITIAL_DELAY = 10  # Initial delay in seconds

    gpt_labels = []
    failed_indices = []

    def fetch_label(i):
        attempt = 0
        question = dataset[i]
        while attempt < MAX_RETRIES:
            try:
                messages = []
                if assistant_message:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_message,
                        }
                    )
                if system_func:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": system_func(question),
                        }
                    )
                if user_func is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": user_func(question),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": question,
                        }
                    )
                completion = client.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    # temperature=0.0,  # to make the output (more) deterministic
                )
                if return_str:
                    label = completion.to_dict()["choices"][0]["message"]["content"]
                else:
                    label = completion.to_dict()["choices"][0]
                # label = completion.to_dict()["choices"][0]["message"]["content"]
                return (i, label, question)
            except OpenAIError as e:
                if "status" in e.body.keys() and e.body["status"] == 400:
                    print(f"Skipping index {i} due to error 400.")
                    return (i, "", question)  # return empty string if error 400
                print(
                    f"Error fetching label for index {i} (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                )
                time.sleep(INITIAL_DELAY * (2**attempt))  # Exponential backoff
                attempt += 1
        return None  # Return None if all retries fail

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(fetch_label, i): i for i in range(len(dataset))
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                label = future.result()
                if label is not None:
                    gpt_labels.append(label)
                    # print(f"Fetched label for index {index} successfully.")
                else:
                    print(
                        f"Failed to fetch label for index {index} after {MAX_RETRIES} attempts."
                    )
                    failed_indices.append(index)
            except Exception as e:
                print(f"Exception for index {index}: {e}")

    for i in failed_indices:
        label = fetch_label(i)
        if label is not None:
            gpt_labels.append(label)

    return gpt_labels


gpt4_label_math = lambda x: gpt4_call(
    x, assistant_message=GPT4_MATH_LABEL_SYSTEM_PROMPT, user_func=None
)
