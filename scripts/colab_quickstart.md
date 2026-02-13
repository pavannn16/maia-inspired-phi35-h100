# Colab Quickstart

## Setup
- `pip install -r requirements.txt`

## Run BF16 offline benchmark
- `python bench/offline_bench.py --config configs/experiment_matrix.yaml --model microsoft/Phi-3.5-mini-instruct --config_id C0 --repeats 1`

## Run lm-eval
- `python eval/lm_eval_runner.py --config configs/experiment_matrix.yaml --model microsoft/Phi-3.5-mini-instruct --out_json results/lm_eval.json`
