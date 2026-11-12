# Wafer Probe Yield Monitoring and Defect Pareto

Yield monitoring and defect-pattern analysis on the WM-811K wafer map dataset, structured like a weekly yield review for a packaging and test program.

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

Fill in from `outputs/summary.json` after running on the real data.
