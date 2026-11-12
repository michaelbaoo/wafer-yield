# Wafer Probe Yield Monitoring and Defect Pareto
# Michael Bao - UCSD B.S. Mathematics, Computer Science

Yield monitoring and defect-pattern analysis on the WM-811K wafer map dataset, structured like a weekly yield review for a packaging and test program.
Full write-up: [reports/wafer_yield_report.pdf](reports/wafer_yield_report.pdf)
## status

| phase | scope | state |
|---|---|---|
| 1 | Load data, wafer and lot yield, label distribution | done |
| 2 | SPC p-chart on lot yield, out-of-control lots | next |
| 3 | Defect Pareto by count and dies lost | planned |
| 4 | Defect pattern classifier | planned |
| 5 | One-page yield review | planned |

## data

WM-811K (LSWMD.pkl) from Kaggle. Each wafer map is a 2D array where 0 is no die, 1 is a passing die, and 2 is a failing die. Place the file at `data/LSWMD.pkl`. It is not committed to the repo.

## definitions

- Wafer yield is passing dies divided by tested dies on one wafer.
- Lot yield is total passing dies divided by total tested dies across the lot. It is die-weighted, so larger wafers count more.
- Wafers with no tested dies are excluded from yield calculations and counted separately.

## run

```
python -m pytest -q
python scripts/run_phase1.py --data data/LSWMD.pkl
```

Outputs go to `outputs/`.

## phase 1 results

- 811,457 wafers across 46,293 lots, about 1.49 billion tested dies
- 172,950 wafers labeled (21.3%) across 9 classes
- Overall die-weighted yield 93.3%, median wafer yield 92.9%, median lot yield 92.4%
- 632 distinct wafer map shapes

## phase 2 results
- Laney p' chart on lot fail rate, lots ordered by lot number (no timestamps in the data, so order is assumed)
- Lot-to-lot variation is 31x binomial noise (sigma_z = 31.3), so a standard p-chart flags 97% of lots and is not usable
- Laney chart flags 1,716 high-fail lots (3.7%) versus about 62 expected by chance
- Those lots hold 24.2% of all dies lost, with 21.1M dies lost beyond the average rate
- Lot size and lot yield are positively correlated (Spearman 0.34), likely from product mix

## phase 3 results
- Edge-Ring is the largest patterned loss (47.9% of patterned dies lost); 4 patterns cover 84% of it
- Patterned labeled wafers hold only 6.1% of all dies lost; unlabeled wafers hold 81.8%
- Labeled and unlabeled yield distributions differ (KS = 0.35), so the Pareto describes the labeled set only
- In Laney-flagged lots, Random (57x), Donut (30x), and Center (7.7x) are over-represented; Scratch is under-represented (0.33x)
- Chronic loss (Edge-Ring) and lot excursions (Random, Donut, Center) point to separate workstreams

## phase 4 results
- Gradient-boosted classifier on 85 engineered features (grid, ring, sector, and cluster statistics), trained with lot-level train/test split
- Held-out lots: 87.4% macro recall, 94.0% accuracy; Edge-Ring 96% recall at 98% precision
- Loc, Scratch, and Edge-Loc have low precision (33-60%) because rare-class weighting trades false alarms for recall
- Labeling all 638,507 unlabeled wafers reranks the patterned loss Pareto: Near-full (37.8%) and Random (17.3%) lead, Edge-Ring falls to third (15.2%)
- The labeled-only Pareto underweights excursion patterns because near-dead wafers were mostly left unlabeled
- Model-free check: wafers below 20% yield are 5.5x more common among unlabeled wafers (2,310 of 638,507) than labeled ones (113 of 172,950)