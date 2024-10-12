import os

# # OPTIONAL: set the HF_HOME environment variable to the directory where the Hugging Face models will be stored
# os.environ['HF_HOME'] = './huggingface'

from dotenv import load_dotenv
import os
import time
import pickle

load_dotenv()  # take environment variables from .env
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError(
        "Hugging Face token not found in environment variables. Please set HF_AUTH_TOKEN."
    )

os.environ["HF_TOKEN"] = token

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

import sys

# Add the directory containing 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import PAD_TOKEN, MAX_LENGTH


def inference(
    model_id: str,
    f_prompt,
    dataset_id: str = "openai/gsm8k",
    lora_version: str = None,
    few_shot: bool = False,
    batch_size: int = 100,
    n_shots: int = 3,
    cot: bool = False,
    save: bool = True,
    output_root: str = "output",
    n_test: int = None,
    quantize: bool = True,
    data_split: str = "test",
    n_train: int = None,
    ds_level: int = None,
    max_new_tokens: int = 1512,
    lora_root: str = "./lora_output",
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = PAD_TOKEN

    # note that this either saves or reads the model weights from /home/azureuser/.cache/huggingface/hub
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=quantize,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if lora_version:
        model.load_adapter(f"{lora_root}/{model_id}/{lora_version}")
    model.eval()

    dataset = load_dataset(dataset_id, "main")
    dataset = dataset[data_split]
    if ds_level:
        dataset = dataset.filter(lambda x: x["level"] == f"Level {ds_level}")
    if n_test:
        dataset = dataset.shuffle(seed=42).select(list(range(n_test)))
    if n_train:
        dataset = dataset.select(list(range(n_train)))
    dataset_chat = f_prompt(dataset)

    def generate_predictions(data, b, tokenizer):
        all_outputs = []
        num_samples = len(data)
        start_time = time.time()

        for start_idx in range(0, num_samples, b):
            end_idx = min(start_idx + b, num_samples)
            batch_inputs = [data[i] for i in range(start_idx, end_idx)]

            batch_inputs = tokenizer.apply_chat_template(batch_inputs, tokenize=False)
            batch_inputs = tokenizer(
                batch_inputs,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(device)

            # greedy decoding
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(decoded_outputs)

            print(f"Processed batch {start_idx // b + 1}/{(num_samples + b - 1) // b}")

        # Print time taken for the whole process
        print(f"Time taken: {time.time() - start_time} seconds")
        return all_outputs

    predictions = generate_predictions(dataset_chat, batch_size, tokenizer)

    # save as pickle file
    file_suffix = ""
    if lora_version:
        file_suffix += f"_lora_{lora_version}"
    elif few_shot:
        file_suffix += f"_few_shot_{n_shots}" if few_shot else ""

    if cot:
        file_suffix += "_cot"
    if not quantize:
        file_suffix += "_non_quantized"

    dataset_id = dataset_id.split("/")[-1]
    _level = f"_level_{ds_level}" if ds_level else ""
    output_dir = (
        f"{output_root}/{model_id}_{dataset_id}{_level}_predictions{file_suffix}.pkl"
    )

    if save:
        with open(output_dir, "wb") as f:
            pickle.dump(predictions, f)

    return predictions, output_dir, file_suffix
