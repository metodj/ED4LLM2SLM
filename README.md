# Environment setup

1. Clone or download this repo. `cd` yourself to it's root directory.
2. Create and activate python [conda](https://www.anaconda.com/) enviromnent: `conda create --name data-distil python=3.8`
3. Activate conda environment:  `conda activate data-distil`
4. Install dependencies, using `pip install -r requirements.txt`
5. Set `huggingface` and `openai` credentials in `.env`

# Lion Data Generation

1. Collect SLM's and LLM's predictions by running: `python generation/seed_inference.py`
2. Run: `python generation/lion/generate.py` (for `slm_preds_path` and `llm_preds_path` use save paths from Step 1.)
    - this will generate the following datasets (`<SAVE_DIR>` is specified in `generate.py` script, `<DATE_TIME>` is an automatically generated data id):
        - `<SAVE_DIR>/<DATE_TIME>/lion_all`
        - `<SAVE_DIR>/<DATE_TIME>/lion_hard`
        - `<SAVE_DIR>/<DATE_TIME>/lion_easy`


# Finetuning and Inference

Supported SLMs:
- [Mistral7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Llama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 

1. Run: `pyhton finetuning/pipeline.py`
    - for finetuning on the Lion data, set `ds_train` param to the path of one of the locally saved Lion datasets, e.g. `ds_train = "data/gsm8k/lion/2024-07-06_18-20-59/lion_hard"` (here `<SAVE_DIR> = "data/gsm8k/lion"` and `<DATE_TIME> = "2024-07-06_18-20-59"`)