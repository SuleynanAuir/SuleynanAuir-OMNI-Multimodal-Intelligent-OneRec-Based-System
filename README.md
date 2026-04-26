<div align="center">


<img src="./assets/logo.png" width="500em" ></img> 

**An Open-Source Framework for
Scaling Generative Recommendation**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
<a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>

<a href="https://arxiv.org/abs/2510.24431">📄 Technical Report</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Huggingface</a> | <a href="https://modelscope.cn/models/k925238839/MiniOneRec">🤖  Modelscope</a>
</div>

**MiniOneRec** is the first fully open-source **generative recommendation** framework, which provides an end-to-end workflow spanning **SID construction**, **supervised fine-tuning (SFT)**, and recommendation-oriented **reinforcement learning (RL)**. 

---

## 📢 Announcement

- 2026-04-26 — We improved training/evaluation observability and stability:
     - SFT progress logging now prints every 100 steps (instead of every step).
     - Added `visualize_training_metrics.py` to plot training `loss`, `grad_norm`, and `learning_rate` from `trainer_state.json`.
     - RL logs now include diagnostic metrics such as `loss_raw`, `loss_abs`, `policy_loss`, `kl_loss`, and advantage statistics.
     - Evaluation now prints real-time progress and supports adaptive OOM retry (auto-reduce batch size / beam size).
     - `evaluate.sh` now uses explicit `MiniOneRec` env Python by default to avoid environment mismatch.

- 2026-01-04 — Regarding the potential discrepancies between the reproduced results based on the Instruct model and our reported metrics, please check whether the CC metric in the evaluation log is non-zero (refer to calc.py). If it is non-zero, it indicates that the model is still generating a large number of invalid items, and constrained decoding has not been successful. We suspect this issue may be related to the versions of dependencies such as the transformer library, and we are still investigating the cause to provide a universal solution. In the meantime, you may switch the Instruct model to a base model, such as Qwen2.5-base, to avoid this problem.

- 2025-12-04 — We update new scripts to support processing the Amazon23 dataset.

- 2025-12-01 — We fix a bug in data.py that could cause the SID–item alignment task to see the answers in advance. This was because we had previously attempted to use partial trajectories to guide the full SID–item generation and does not affect the model performance.

- 2025-11-20 — The SID construction method in **RQ-Kmeans+** has been updated (first proposed in **GPR** and this is the first open-source reproduction).

- 2025-11-19 — We implemented a multi-GPU parallel text-to-embedding method based on Accelerate, which is significantly more efficient than the original version: rq/text2emb/amazon_text2emb.py

- 2025-11-19 — The SID construction method in **constrained-RQ-Kmeans** has been updated.

- 2025-11-07 — Thank you for submitting issues! Based on your feedback, we have released a new implementation. If you encounter any problems while running the code, please update to and consult the **latest version** first.
  
- 2025-11-07 — You can now choose to freeze the LLM parameters during the SFT stage and train only the embeddings for the newly added SID vocabulary.

- 2025-10-31 — You can now directly download the implementation **checkpoints** of our MiniOnRec model.

- 2025-10-31 — The SID construction method in **RQ-Kmeans** has been updated.

---

## 🛠️ Key Techniques 
<div align="center">
<img src="./assets/minionerec_framework.png" width=100% ></img> 
</div>

- **SID Construction: MiniOneRec begins by transforming every product into a compact, semantically meaningful token.** It concatenates an item’s title and description, feeds this sentence through a frozen text encoder, and then quantises the resulting embedding with a three-level RQ-VAE.

- **SFT: With all items rewritten as SIDs, the model is first trained in a supervised fashion.** It views the chronologically ordered user history as a token sequence and learns, via next-token prediction, to generate the SID of the next product the user is likely to consume. Crucially, this stage is co-trained with a set of language-alignment objectives that map back and forth between natural language and SID space, allowing the recommender to inherit the world knowledge embedded in large language models while grounding that knowledge in discrete item codes.

- **Recommendation-Oriented RL: After SFT, MiniOneRec is further polished with a recommendation-oriented RL phase based on GRPO.** Multiple candidate recommendations are generated for each prompt, their rewards are normalised within the group to stabilise gradients, and a KL penalty keeps the updated policy close to its reference. Because the action space is a closed list of item SIDs, the system switches to constrained beam search, which guarantees that every beam is unique and valid, greatly improving sampling efficiency and diversity. The reward signal itself blends a binary correctness term with a rank-aware component that penalises high-probability yet incorrect items more heavily, and can be augmented with collaborative-filtering scores. Together, this pipeline enables MiniOneRec to couple dense linguistic knowledge, achieving a high-performance, lightweight generative recommendation system.

---

## 📊 Evaluation

<div align="center">
<img src="./assets/minionerec_main_result.png" width=100% ></img> 
</div>

---

## 🗂️ Repository Overview

| File / Directory          | Description                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `sft.sh`                  | Shell script to start the Supervised Fine-Tuning (SFT) stage                                           |
| `sft.py`                  | Python implementation of the SFT training loop                                                            |
| `sft_gpr.py`              | GPR-inspired SFT with Value-Aware Fine-Tuning (VAFT): implements weighted loss based on simulated item value                            |
| `rl.sh`                   | Shell script to start the Reinforcement Learning (RL) stage                             |
| `rl.py`                   | Python implementation of the RL training loop                                              |
| `rl_gpr.py`               | GPR-inspired RL with Hierarchy Enhanced Policy Optimization (HEPO)                                                 |
| `minionerec_trainer.py`   | MiniOneRec trainer — GRPO-based trainer specialized for generative recommendation                              |
| `configs/`                | YAML configuration files                                            |
| `evaluate.sh`     | One-click offline Top-K evaluation script                                                        |
| `evaluate.py`     | Evaluation utilities for computing HR@K and NDCG@K.                                                           |
| `visualize_training_metrics.py` | Training metrics visualizer (reads `trainer_state.json` and plots `loss` / `grad_norm` / `learning_rate`) |
| `LogitProcessor.py`                | Logit processor for constrained decoding (Python implementation)                                         |
| `data.py`                | Data pipeline for SFT and RL training                          |
| `convert_dataset.py`                | Converts an RQ-trained dataset to the SFT-then-RL format                                            |
| `convert_dataset_gpr.py`           | GPR-inspired dataset converter: injects simulated heterogeneous tokens (U/E/I/O) to emulate unified input representation                                         |
| `data/amazon18_data_process.sh`                |    Shell script to filter and preprocess Amazon18 data into an RQ-ready format                                      |
| `data/amazon18_data_process.py`                |   Python implementation of the Amazon18 data preprocessing pipeline                                        |
| `data/amazon18_data_process_gpr.py`            |   GPR-inspired Amazon18 preprocessing: extracts heterogeneous features for unified input representation                         |
| `data/amazon23_data_process.sh`                |    Shell script to filter and preprocess Amazon23 data into an RQ-ready format                                      |
| `data/amazon23_data_process.py`                |   Python implementation of the Amazon23 data preprocessing pipeline                                        |
| `rq/text2emb/amazon_text2emb.sh`                |   Shell script to generate item embeddings (title + description) via emb_model for the Amazon dataset                                   |
| `rq/text2emb/amazon_text2emb.py`                |   Python implementation of the above embedding generation                                         |
| `rq/text2emb/amazon_text2emb_gpr.py`           |   GPR-inspired text-to-embedding                                 |
| `rq/generate_indices.py`                |   Generates the SID file after training an RQ-VAE model                                       |
| `rq/rqvae.sh`                |   Shell script to train RQ-VAE on Amazon item embeddings                        |
| `rq/rqvae.py`                |   Python implementation of RQ-VAE training                                            |
| `rq/rqkmeans_faiss.py`                |   Python implementation of RQ-Kmeans training based on faiss                                          |
| `rq/rqkmeans_constrained.py`                |   Python implementation of Constrained RQ-Kmeans                         |
| `rq/rqkmeans_constrained.sh`                |   Shell script to train constrained RQ-Kmeans constrained on Amazon item embeddings                        |
| `rq/rqkmeans_plus.py`                |   Python implementation of RQ-Kmeans+                        |
| `rq/rqkmeans_plus.sh`                |   Shell script to train RQ-Kmeans+ constrained on Amazon item embeddings                        |
| `rq/generate_indices_plus.py`                |   Generates the SID file after training an RQ-Kmeans+ model                                       |
| `rq/generate_indices_plus.sh`                |   Shell script to generate the SID file after training an RQ-Kmeans+ model                                       |
| `requirements.txt`        | List of Python dependencies                                                                                |

---

## 🚀 Quickstart

Use the pre-trained Industrial/Office SIDs we provide for a quick start!
Reproduction can be achieved with just 4–8 A100/H100 GPUs.

### 1. Create an isolated Python environment

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. SFT (Supervised Fine-Tuning)

Train the model using supervised fine-tuning on the SID-to-target-item prediction task.

```bash
bash sft.sh
```

**Logging behavior:**
- SFT progress callback prints every `100` steps (and final step), reducing noisy logs.

**Key Parameters (in `sft.sh`):**
- `base_model`: Path to your base LLM (e.g., Qwen or Llama)
- `batch_size`: Batch size per GPU (e.g., 1024)
- `micro_batch_size`: Gradient accumulation micro-batch (e.g., 16)
- `train_file`: Path to training CSV (e.g., `./data/Amazon/train/Industrial_and_Scientific_*.csv`)
- `eval_file`: Path to validation CSV
- `output_dir`: Where to save SFT checkpoints
- `sid_index_path`: Pre-computed SID index JSON (e.g., `./data/Amazon/index/Industrial_and_Scientific.index.json`)
- `item_meta_path`: Item metadata JSON (e.g., `./data/Amazon/index/Industrial_and_Scientific.item.json`)
- `freeze_LLM`: Whether to freeze LLM parameters and only train embeddings (default: False)

**To customize:** Edit `sft.sh` to update paths, model name, and hyperparameters for your setup.

### 4. Recommendation-Oriented RL

Further refine the SFT model using reinforcement learning with group-relative policy optimization (GRPO).

```bash
bash rl.sh
```

**Logging behavior:**
- RL logs include diagnostic fields: `loss_raw`, `loss_abs`, `policy_loss`, `kl_loss`, `adv_mean`, `adv_abs_mean`, `adv_std`.
- In GRPO, `loss` can be displayed as `0.0` due to scale/formatting, while `loss_raw` is non-zero (this is expected).

**Key Parameters (in `rl.sh`):**
- `model_path`: Path to the SFT checkpoint from step 3
- `train_batch_size`: Batch size (e.g., 64)
- `eval_batch_size`: Evaluation batch size (e.g., 128)
- `num_train_epochs`: Number of RL training epochs (e.g., 2)
- `train_file` / `eval_file` / `info_file`: Same as SFT
- `num_generations`: Number of candidates per prompt (e.g., 16)
- `reward_type`: Reward signal type (e.g., `ranking` for rank-aware rewards)
- `learning_rate`: RL learning rate (e.g., 1e-5)
- `beta`: KL penalty coefficient (e.g., 1e-3)
- `beam_search`: Use constrained beam search (configurable in `rl.sh`; default now favors stable exploration)
- `output_dir`: Where to save RL checkpoints

**To customize:** Edit `rl.sh` to update checkpoint paths and hyperparameters.

### 5. Run the evaluation bash

Evaluate the trained model on the test set using offline metrics and advanced analysis figures.

```bash
bash evaluate.sh
```

**Recommended run (with log safety):**
```bash
set -o pipefail
bash evaluate.sh | tee logs/eval_$(date +%F_%H-%M-%S).log
```

**Evaluation Flow:**
1. **Data Splitting**: Distribute test data across available GPUs
2. **Parallel Inference**: Run predictions on each GPU independently
3. **Result Merging**: Aggregate results from all GPUs
4. **Metric Computation**: Calculate HR@K and NDCG@K
5. **Advanced Analysis**: Compute multiple recommendation metrics and generate publication-style figures

**Key Parameters (in `evaluate.sh`):**
- `exp_name`: Path to your trained model checkpoint
- `test_file`: Test CSV (e.g., `./data/Amazon/test/Industrial_and_Scientific_*11.csv`)
- `info_file`: Category info file
- `EVAL_BATCH_SIZE`: Inference batch size (default: `2`)
- `EVAL_NUM_BEAMS`: Beam size for generation (default: `20`)
- `EVAL_MAX_NEW_TOKENS`: Maximum tokens to generate (default: `128`)
- `temperature`: Sampling temperature (default: 1.0)
- `cudalist`: GPU IDs to use (default: 0–7)

**Runtime behaviors (recent updates):**
- Real-time progress logs: `evaluate.sh` prints process start, `evaluate.py` prints stage and batch progress.
- Adaptive OOM retry: if CUDA OOM happens in generation, evaluation auto-reduces batch size, then beam size, and retries.
- Environment stability: `evaluate.sh` uses explicit Python from `MiniOneRec` env by default (`ENV_PYTHON`).

**Output:**
- `./results/{model_name}/final_result_{category}.json`: merged prediction results
- `./results/{model_name}/analysis_{category}/metrics_summary.json`: multi-metric summary
- `./results/{model_name}/analysis_{category}/metrics_table.csv`: per-K metric table
- `./results/{model_name}/analysis_{category}/top1_frequency.csv`: Top-1 prediction frequency statistics
- `./results/{model_name}/analysis_{category}/metrics/`: detailed tables (`metrics_correlation.csv`, `all_prediction_frequency.csv`, etc.)
- `./results/{model_name}/analysis_{category}/figures/ranking/`: ranking figures (`01_ranking_quality_curves_ci95.png`, `06_precision_recall_f1_map.png`, `07_accuracy_coverage_tradeoff.png`)
- `./results/{model_name}/analysis_{category}/figures/quality/`: quality/safety figures (`02_quality_safety_curves.png`, `05_novelty_popularity_bias.png`)
- `./results/{model_name}/analysis_{category}/figures/distribution/`: distribution figures (`03_long_tail_distribution.png`, `04_first_hit_rank_distribution.png`, `08_top1_frequency_bar_top20.png`, `09_lorenz_curve.png`)
- `./results/{model_name}/analysis_{category}/figures/diagnostics/`: diagnostic figures (`10_metric_correlation_heatmap.png`, `11_entropy_gini_curves.png`)
- `./results/{model_name}/analysis_{category}/manifest.json`: generated artifact manifest for reproducible reporting

---

## 📜 Full Pipeline Walk-through

### 0. Prerequisites
- GPUs: <e.g., 4–8 × A100/H100 80 GB or comparable>
- Python: 3.11

### 1. Environment Setup
- **1.1 Clone the repo**
```
git clone https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
```
- **1.2 Create and activate a conda env**
```
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```
- **1.3 Install dependencies**
```
pip install -r requirements.txt
```

### 2. Data Preparation

- **2.1 Download the raw dataset (Optional)**  
  Get it from the official page:
  [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), 
  [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), 
  [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
  Note: The Industrial and Office datasets are included in Amazon 2018; the Amazon 2014 and 2023 versions require slight modifications to our data/amazon18_data_process.py.
- **2.2 Filter and preprocess**
```
bash data/amazon18_data_process.sh \
     --dataset  your_dataset_type \ # e.g. Industrial
     --user_k 5 \
     --item_k 5 \
     --st_year 2017 \
     --st_month 10 \
     --ed_year 2018 \
     --ed_month 11 \
     --output_path ./data/Amazon18
```
- **2.3 Encode item text to embeddings**
```
bash rq/amazon_text2emb.sh \
     --dataset your_dataset_type \ # e.g., Industrial 
     --root your_processed_dataset_path \
     --plm_name qwen \
     --plm_checkpoint your_emb_model_path
```

### 3. SID Construction

Choose either 3.1.1, 3.1.2, 3.1.3 or 3.1.4.

- **3.1.1 Train RQ-VAE on the embeddings**
```
bash rq/rqvae.sh \
      --data_path xxx/data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
```

- **3.1.2 Train RQ-Kmeans on the embeddings**

```
conda install faiss-gpu
python rqkmeans_faiss.py --dataset Industrial_and_Scientific # The RQ-Kmeans method based on semantic embeddings has a relatively high collision rate.
```

- **3.1.3 Train constrained RQ-Kmeans on the embeddings**
For conflicting items, we add an extra layer to perform deduplication; meanwhile, we use a balanced constraint to ensure that the SIDs are evenly distributed.
```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
```

- **3.1.4 Train RQ-Kmeans+ on the embeddings**
```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
bash rqkmeans_plus.sh
```

- **3.2 Generate indices(only RQ-VAE & RQ-Kmeans+ needed)**
```
python rq/generate_indices.py
# or
bash rq/generate_indices_plus.sh
```

- **3.3 Convert dataset format**
```
python convert_dataset.py \
     --dataset_name Industrial_and_Scientific \
     --data_dir /path/to/Industrial_and_Scientific \
     --output_dir /path/to/ourput_dir \

```

### 4. SFT

```
bash sft.sh \
     --base_model your_model_path \
     --output_dir your_ourput_dir \
     --sid_index_path your_.index.json_path \
     --item_meta_path your_.item.json_path
```

### 5. Recommendation-Oriented RL
> (Optional) For production-scale datasets, considering the cost of reinforcement learning and diminishing marginal returns, you can perform the RL stage using only a relatively small subset on the order of tens of thousands of samples.
```
bash rl.sh \
     --model_path your_model_path \
     --output_dir output_dir \
```

### 6. Offline Evaluation

```
bash evaluate.sh \
     --exp_name your_model_path 
```

For low-memory GPUs, start with:
```bash
export EVAL_BATCH_SIZE=1
export EVAL_NUM_BEAMS=10
export EVAL_MAX_NEW_TOKENS=96
bash evaluate.sh
```

### 7. Training Metrics Visualization

Generate local training curves (`loss`, `grad_norm`, `learning_rate`) from a training directory or a specific `trainer_state.json`:

```bash
python visualize_training_metrics.py --path output_dir/xxx
```

Output files:
- `training_analysis/training_metrics.png`
- `training_analysis/training_metrics.csv`

---

## 🩺 Troubleshooting

- **`loss` appears as `0.0` during RL**
     - Check `loss_raw` / `loss_abs` in logs. In GRPO this is often a display-scale effect, not stalled training.

- **Many `No valid tokens found` warnings**
     - This comes from constrained decoding when a beam drifts out of valid prefix paths. Current code includes warning throttling and fallback behavior.

- **`ModuleNotFoundError: transformers` in evaluation**
     - Ensure `MiniOneRec` env is used. `evaluate.sh` now resolves Python via `ENV_PYTHON` (default points to `.../envs/MiniOneRec/bin/python`).

- **CUDA OOM during evaluation**
     - Lower `EVAL_BATCH_SIZE`, `EVAL_NUM_BEAMS`, and/or `EVAL_MAX_NEW_TOKENS`.
     - Use defaults from the updated `evaluate.sh` or the low-memory preset above.

---

## 🤖 Supported LLM Providers

MiniOneRec supports multiple LLM providers for text enrichment tasks (e.g., user preference and item characteristic extraction). Configure the provider in your `api_info` dictionary:

| Provider | `provider` value | Default Base URL | Example Models |
|----------|-----------------|------------------|----------------|
| OpenAI | `"openai"` | — | `text-davinci-003` |
| DeepSeek | `"deepseek"` | `https://api.deepseek.com` | `deepseek-chat` |
| [MiniMax](https://www.minimaxi.com) | `"minimax"` | `https://api.minimax.io/v1` | `MiniMax-M2.7`, `MiniMax-M2.5` |

**Example — using MiniMax:**

```python
api_info = {
    "provider": "minimax",
    "api_key_list": ["your-minimax-api-key"],
    "base_url": "https://api.minimax.io/v1",  # optional, this is the default
}
get_res_batch("MiniMax-M2.7", prompt_list, max_tokens=512, api_info=api_info)
```

### `.env` Auto Loading (Optional)

`rq/text2emb/utils.py` now supports automatic loading from a project-root `.env` file.

1. Copy and edit:
```bash
cp .env.example .env
```

2. Fill your keys in `.env` (example):
```env
LLM_PROVIDER=minimax
API_KEY_LIST=
MINIMAX_API_KEY=your-minimax-api-key
MINIMAX_BASE_URL=https://api.minimax.io/v1
TEMPERATURE=0.4
```

3. Call `get_res_batch` with or without explicit `api_info`:
```python
get_res_batch("MiniMax-M2.7", prompt_list, max_tokens=512, api_info=None)
```

Priority order is: explicit `api_info` > process environment variables > `.env` file.

---

## 📝 Upcoming Features

We are actively extending MiniOneRec’s capabilities. The following enhancements are already on our roadmap:
* ⏱️ **More SID Construction Algorithms**: forthcoming support for R-VQ, RQ-Kmeans, RQ-OPQ, and RQ-VAE-v2 (PLUM).
* ⚙️ **MiniOneRec-Think**: a module that seamlessly integrates dialogue, reasoning, and personalized recommendation, providing an all-in-one solution for complex interactive scenarios.
* 🔍 **Broader Dataset Support**: additional popular public datasets, including Yelp, to further validate the generality of our algorithms.

---

## 🏫 Institutions  <!-- omit in toc -->

This project is developed by the following institutions:

- <img src="assets/lds.png" width="28px"> [LDS](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/index.html)
- <img src="assets/alphalab.jpg" width="28px"> [AlphaLab](https://alphalab-ustc.github.io/index.html)
- <img src="assets/next.jpg" width="28px"> [NExT](https://www.nextcenter.org/)
 
---

## 🧩 Contributing

We welcome and appreciate all contributions! If you have ideas to improve MiniOneRec, please feel free to submit a pull request (PR).

---
## 🙏 Acknowledgements

This repository reuses or adapts portions of code from the following open-source projects. We gratefully acknowledge their authors and contributors:

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

---

## 🔖 Citation <!-- omit in toc -->

If you find our code/paper/model helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}

```

---

<div align="center">
We welcome contributions from the community! 🤝
</div>
