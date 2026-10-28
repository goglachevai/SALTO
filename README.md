# SALTO (Snippet and Autoencoder-based Labeling of Time series coming Online)

This repository is related to the SALTO (Snippet and Autoencoder-based Labeling of Time series coming Online) algorithm that classifies subsequences of a streaming time series with a neural network. SALTO is authored by Andrey Goglachev (goglachevai@susu.ru) and Mikhail Zymbler (mzym@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the SALTO source code (in Python).

SALTO is trained on the output of **PaSTiLa** (the `results/` directory): the snippets found by PaSTiLa and the labeling of the series. PaSTiLa itself is not called. Each subsequence of length m_opt is passed through an autoencoder (the extractor), which reconstructs the snippet it belongs to. The subsequence gets the label of the snippet closest to the reconstruction. Two distances are supported: C22dist (`--metric c22`, as in PDSS) and MPdist (`--metric mpdist`, as in PSF).

## Usage

The commands below use `py` (Windows) and are run from the SALTO root directory. On Linux/macOS, use `python3`.

Install dependencies:

```
py -m pip install -r requirements.txt
```

Train:

```
py src/train.py --series datasets/MixedBag/GreatBarbet1F_Jog_1800_200.txt --results results --out src/models/salto.pt
```

| Parameter             | Description                                                                  |
|-----------------------|------------------------------------------------------------------------------|
| `--series`            | Text file containing the time series (default: `datasets/MixedBag/GreatBarbet1F_Jog_1800_200.txt`) |
| `--results`           | PaSTiLa output directory for this series (default: `results`)               |
| `--out`               | Model file to write (default: `src/models/salto.pt`)                        |
| `--epochs`            | Maximum number of epochs (default: 300)                                     |
| `--batch`             | Batch size (default: 32)                                                    |
| `--lr`                | Learning rate (default: 3e-3)                                               |
| `--gamma`             | Exponential learning rate decay per epoch (default: 0.99)                   |
| `--patience`          | Early stopping: epochs without val RMSE improvement (default: 30)           |
| `--block`             | Length of the blocks held out for val/test (default: 2·m)                   |
| `--full`              | Train on the whole series, without held-out blocks                          |
| `--no-balance`        | Sample windows uniformly instead of drawing classes with equal probability  |
| `--metric`            | Distance used to pick the nearest snippet: `c22` (C22dist) or `mpdist` (MPdist) |
| `--mp-window`         | Length of the subsequences inside MPdist (default: m/2, as in PSF)          |
| `--seed`              | Random seed (default: 0)                                                    |
| `--device`            | `cuda` or `cpu` (default: `cuda` if available)                              |

Evaluate:

```
py src/evaluate.py --model src/models/salto.pt --change-points 1800
```

| Parameter             | Description                                                                  |
|-----------------------|------------------------------------------------------------------------------|
| `--model`             | Trained model (default: `src/models/salto.pt`)                              |
| `--series`, `--results` | Same as for `train.py`                                                    |
| `--change-points`     | Ground-truth change points of the series, to compare against the true labeling (optional) |
| `--metric`, `--mp-window` | Override the distance stored in the model (the extractor does not depend on it). `c22` needs a model trained with `--metric c22` |
| `--devices`           | Devices that make the inference (default: `cpu` and, if available, `cuda`) |

Streaming classification (with `src/` on the import path):

```python
from salto import Salto
stream = Salto.load("src/models/salto.pt", device="cuda").stream()
label = stream.push(value)   # None until m_opt points have arrived
```

The `results/` directory must contain the PaSTiLa output for the best segment length m_opt: `snippets.txt` (K × m), `indicies.txt` (zero-based snippet positions), `labels.txt` (n − m + 1 labels in 0..K−1; PSF writes n − m, the last window is then skipped), `fracs.txt`, `profiles.txt` (K × (n − m + 1)).

## Output

`train.py` creates the following files:

- `<out>` (e.g. `salto.pt`): the extractor weights, the snippets, the distance (`c22` or `mpdist`, with the MPdist window) and, for C22dist, the feature normalization parameters. `Salto.load` needs only this file.
- `<out>_metrics.json` (e.g. `salto_metrics.json`): training time, best val RMSE, and accuracy and macro F1 against the PaSTiLa labeling on train, val, test and the whole series.

## Citation

```
@article{Goglachev2024,
 author    = {А.И. Гоглачев},
 title     = {Классификация потокового временного ряда на основе нейросетевых технологий и поведенческих шаблонов},
 journal   = {Вестник ЮУрГУ. Серия: Вычислительная математика и информатика},
 volume    = {13},
 number    = {3},
 pages     = {79-94},
 year      = {2024},
 doi       = {10.14529/cmse240305},
 url       = {https://doi.org/10.14529/cmse240305}
}
```
