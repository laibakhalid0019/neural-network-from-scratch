# Deep Learning for Perception - Assignment 1
### Building, Breaking and Fixing a Neural Network (Fashion-MNIST, PyTorch, Kaggle T4 x2)

This repository contains the complete implementation for DLP Assignment 1: a from-scratch NumPy
MLP with a PyTorch gradient check (Part 1), an activation-function study (Part 2), a loss-function
comparison plus a regression side-experiment (Part 3), an optimizer comparison (Part 4), a
deliberately overfit model (Part 5), a seven-method regularisation study (Part 6), and a
random-search + 5-fold cross-validation hyperparameter search ending in a single held-out
test-set evaluation (Part 7).

## Final result

| | |
|---|---|
| **Test accuracy** | 88.09% |
| **Macro precision / recall / F1** | 88.16% / 88.09% / 88.00% |
| **Selected configuration** | learning rate = 0.00222, hidden width = 128, dropout = 0.336 |
| **Best regularisation method (Part 6)** | Dropout, p = 0.6 (generalisation gap 16.57 → 6.05 points) |

Full part-by-part results are in `DLP_Assignment01_Summary.docx`.

## Repository contents

```
.
├── DL_ASS01_XXF_YYYY.ipynb          # the full notebook, all 7 parts, executed with outputs
├── DLP_Assignment01_Summary.docx    # one-page results summary
├── DLP_Assignment01_Guide.md        # part-by-part explanation, roadmap, requirements checklist
└── README.md                        # this file
```

## How to reproduce

1. **Platform:** [Kaggle Notebooks](https://www.kaggle.com/), accelerator set to **GPU T4 x2**.
2. **Dataset:** attach the Fashion-MNIST dataset via *Add Input* → search "Fashion MNIST" →
   [`zalando-research/fashionmnist`](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
   The notebook auto-detects the CSV path under `/kaggle/input/`, so no path needs to be edited
   as long as the dataset is attached.
3. **Internet:** turn Internet **ON** in the notebook's settings (needed once, for Part 3's
   `sklearn.datasets.fetch_california_housing` regression dataset).
4. Open `DL_ASS01_XXF_YYYY.ipynb` and run **Kernel → Restart & Run All**. Do not run cells out of
   order — later cells in every part reuse variables, loaders, and models created by earlier
   cells in that same part.
5. Expected total runtime on a single T4: Parts 1–5 are quick (a few minutes total); Part 6 runs
   seven full training passes on the 2,000-sample overfitting model; **Part 7 is the longest step**
   (a 12-configuration × 5-fold cross-validation random search, ~60 short training runs) and
   dominates the total wall-clock time.
6. Every result printed in `DLP_Assignment01_Summary.docx` was read directly from an executed run
   of this notebook with `SEED = 42` set at the top — re-running end-to-end should reproduce the
   same numbers (small floating-point differences aside).

## Reproducibility notes

- A single global seed (`SEED = 42`) is set for `random`, `numpy`, and `torch` (CPU and all GPUs),
  and `torch.backends.cudnn.deterministic = True` / `benchmark = False` are both set — both are
  required together for repeatable results on GPU.
- The 60,000-row Fashion-MNIST training pool is split 80/20 (48,000 train / 12,000 validation)
  with a fixed `random_state`; the 10,000-row test set is not touched by any model-selection step
  and is evaluated exactly once, in Part 7's final cell.
- Part 7's cross-validation search runs on a 12,000-sample subsample of the training pool for
  compute-budget reasons (12 configs × 5 folds is otherwise expensive on a single GPU session);
  this is disclosed directly in that section of the notebook. The selected configuration is
  retrained on the full 60,000-sample training pool before the final test evaluation.

## Environment

- Python 3.12, PyTorch (CUDA build), NumPy, pandas, scikit-learn, matplotlib, scipy.
- No packages beyond Kaggle's default notebook image and `torch`/`torchvision` are required.
