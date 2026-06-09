# Lung Nodule Detection — Local Fine-tuning (MONAI / Apple Silicon)

Fine-tuning a LUNA16-pretrained RetinaNet (MONAI) on local clinic CT data,
run on Apple Silicon (M4 Max / MPS). This is the fine-tuning + evaluation
portion of a university capstone project; pretraining was done separately.

## Privacy note
This repository contains **code, configuration, and aggregate results only**.
It does **not** include patient CT images, real annotations, or model weights,
because they are sensitive clinical data. `annotations_format_example.json`
shows the expected data structure with synthetic values.

## Contents
```
code/         Fine-tuning + evaluation scripts
config/       Hyperparameters + path template (edit <PATH_TO>)
diagnostics/  Coordinate-system + annotation QA tooling
results/      FROC curve + training log (no patient data)
```

## What it does
- Fine-tunes a pretrained RetinaNet detector on local CT scans (domain adaptation).
- Runs on Apple Silicon (MPS): AMP disabled, gradient accumulation, per-epoch
  cache management.
- Evaluates with FROC (sensitivity at fixed false-positives per scan).
- Includes a per-scan coordinate-alignment diagnostic (LPS vs RAS), which is
  essential before trusting any metric on multi-source clinical data.

## Setup
Python 3.11, PyTorch (MPS or CPU), MONAI, numpy, matplotlib, tensorboard,
warmup_scheduler.

1. Provide your own CT scans (NIfTI) and a datalist (see
   `annotations_format_example.json` for the format).
2. Edit `config/environment.json`: replace every `<PATH_TO>` and point
   `pretrained_model_path` at your pretrained weights.

## Run
```bash
python code/luna16_finetune_mps.py -e config/environment.json -c config/config_train.json
tensorboard --logdir <PATH_TO>/lung_project/runs
```

## Result (local validation)
- FROC mean sensitivity: 0.75 (10 scans / 24 nodules — a feasibility baseline,
  not a benchmark). See `results/`.

## Limitations
- Fine-tuning adapted the model to the local domain but did not reduce the
  false-positive rate (limited by data volume and annotation completeness).
- Annotations were reviewed from model outputs, not by a radiologist.
- Small validation set; results are a trend, not a stable benchmark.

## License
[ADD a license — e.g. MIT — or note "for academic coursework". Confirm with
your teammate whose repo this is, and with your course, before publishing.]
