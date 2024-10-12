# Rejection Sampling

1. First generate synthetic dataset (e.g., using Lion: `python generation/lion/generate.py`)
    - get the path of the saved dataset
2. Filter synthetic dataset using either `self_consistency.py` or `self_verification.py`
    - use the path of the saved synthetic dataset from step 1
    - get the path of the filtered dataset
3. Finetune the SLM on the filtered synthetic data
    - use the path of the filtered dataset (as `ds_train`) from step 2

