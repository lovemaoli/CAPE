# CAPE: Co-occurrence-Aware Pool Expansion

CAPE is a training-free plug-in for generative recommenders with semantic item IDs.
The generator, its tokenizer, its beam and its checkpoints stay unchanged. CAPE only
changes the candidate pool and the order of the returned list:

1. **Memorization channel (retrieval).** From the training split only, `U[j,c]` counts
   the distinct users who interacted with both `j` and `c`. Support
   `S_u(c) = sum_p U[i_p, c]` ranks the tail items outside the 50-item beam; the top
   25 join the pool. Zero-support items fill any remaining slots in catalog-index order.
2. **Calibrated ranking.** `W[j,c]` counts co-occurrence events at distinct positions
   of the training sequences. Evidence
   `e_u(c) = 1[c in tail] * sum_p W[i_p, c] / max(1, n)`, and every candidate in the
   75-item pool is scored by `N(ll) + lambda * N(e)`, where `N` is per-pool min-max
   normalization with `eps = 1e-12` and `ll` is the frozen generator's original-vocabulary
   log-likelihood. A constant evidence vector maps to zero.
3. **Top-1 anchoring.** The backbone's first item stays at rank 1; the remaining items
   follow the fused score, with ties broken by pool index.

The tail is the 80% least frequent items among those observed in training (stable
first-seen order breaks frequency ties). Items that appear only in the catalog info file
remain valid candidates but are not tail items.

This repository contains **only CAPE**. Baselines, beam search, model training,
identifier construction and frozen checkpoints belong to the backbone projects
(MiniOneRec and TIGER) and are not redistributed here.

## Repository layout

```
cape/                     CAPE library
  data.py                 SID-level CSV/JSONL readers
  statistics.py           train-only popularity, tail split, U and W
  rerank.py               memorization channel, score fusion, top-1 anchoring
  metrics.py              HR/NDCG, paired user-cluster bootstrap, lambda selection
scripts/
  download_data.py        download public raw datasets
  prepare_data.py         build the cache from SID-level CSV splits
  run_cape.py             evaluate CAPE on pre-computed beam outputs
configs/
  data_sources.json       public dataset URLs
  lambda.json             fusion weights selected on validation
```

## Setup

Python 3.9+ with NumPy.

```bash
pip install -r requirements.txt
```

Run all commands from the repository root.

## Data

No dataset, catalog split, statistics cache or checkpoint is distributed here.

1. Download the public raw data:

   ```bash
   python scripts/download_data.py --list
   python scripts/download_data.py --all
   ```

   The Amazon Reviews 2018 5-core files come from the McAuley Lab at UCSD. LastFM-1K
   (Celma, 2010) needs to obtain the listening log from its
   official distribution and use `python scripts/download_data.py --url <URL> --name LastFM1K`.

2. Produce the **SID-level CSV splits** with the MiniOneRec preprocessing pipeline
   (5-core filtering, chronological 8:1:1 split, histories of the latest ten items and
   semantic-ID construction). Each split is a CSV with columns `user_id`,
   `history_item_sid` (a stringified list of SIDs) and `item_sid`. The matching info
   file is tab-separated with the SID in the first column.

3. Build the CAPE statistics cache:

   ```bash
   python scripts/prepare_data.py \
       --name Industrial_and_Scientific \
       --train data/processed/Industrial_and_Scientific/train.csv \
       --info  data/processed/Industrial_and_Scientific/info.txt \
       --valid data/processed/Industrial_and_Scientific/valid.csv \
       --test  data/processed/Industrial_and_Scientific/test.csv
   ```

   This writes `data/processed/Industrial_and_Scientific/stats.npz` (popularity, tail
   mask, U and W) and prints catalog/tail counts for comparison with the paper.

## Input format for `run_cape.py`

`--test` (and optionally `--valid`) is a JSON array or JSONL file with one record per
request. The frozen backbone must already have produced the beam and the teacher-forced
likelihoods; CAPE does not call the model itself.

```json
{
  "sample_id": 0,
  "user_id": "A1",
  "history": ["<sid_a>", "<sid_b>"],
  "target": "<sid_t>",
  "candidates": ["<sid_1>", "<sid_2>", "<sid_3>"],
  "ll": [-1.2, -1.5, -1.8],
  "anchor": "<sid_1>",
  "extra_ll": {"<sid_x>": -3.1, "<sid_y>": -4.7}
}
```

## Evaluation

```bash
# Use a published lambda
python scripts/run_cape.py --stats data/processed/.../stats.npz \
    --test data/processed/.../test.jsonl \
    --case MiniOneRec_Office_Products_seed42 \
    --out results/rankings.jsonl

# Or re-select lambda on the full validation split
python scripts/run_cape.py --stats ... --valid valid.jsonl --test test.jsonl \
    --select-lambda --bootstrap
```

The script reports overall and tail HR@5/10, NDCG@10, the paired user-cluster
bootstrap intervals, and writes the per-request rankings. `configs/lambda.json`
lists the weights selected on validation for the six frozen-model cases in the paper

## Citation

```bibtex
@article{cape2026,
   title={Generative Recommenders Forget the Long Tail: Restoring Memorization in Retrieval and Ranking}, 
   author={Xu Huang and Jiancan Wu and Hongzheng Qin and Jiawei Feng and Can Ye and Wencong Zeng and Fengze Li and Xiang Wang},
   year={2026},
   eprint={2610.29298},
   archivePrefix={arXiv},
   primaryClass={cs.IR},
   url={https://arxiv.org/abs/2610.29298}, 
}
```
