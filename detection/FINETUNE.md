# Fine-tuning LUNA16 Detection Model

This note describes a practical workflow for fine-tuning the MONAI RetinaNet lung nodule detector on local clinic CT data.

## 1. Data Split

Split local data at the patient level:

```text
70% training / 15% validation / 15% test
```

For smaller datasets:

```text
80% training / 10% validation / 10% test
```

Do not let scans from the same patient appear in multiple splits. Keep the test set untouched until final evaluation.

## 2. Datalist Format

Use the same datalist structure as LUNA16:

```json
{
  "training": [
    {
      "image": "case001/case001.nii.gz",
      "box": [[100.0, 22.0, -147.0, 8.8, 8.8, 8.8]],
      "label": [0]
    }
  ],
  "validation": [
    {
      "image": "case101/case101.nii.gz",
      "box": [[-55.3, 4.0, -154.7, 6.0, 6.0, 6.0]],
      "label": [0]
    }
  ]
}
```

The box format must match `gt_box_mode` in the config. For the default LUNA16 config, this is:

```json
"gt_box_mode": "cccwhd"
```

This means center coordinate plus width, height, and depth in world coordinates.

## 3. Environment File

Create a local environment file, for example `config/environment_local_finetune.json`:

```json
{
  "data_base_dir": "/path/to/local/resampled/images",
  "data_list_file_path": "/path/to/local_datalist.json",
  "model_path": "./trained_models/model_local_finetuned.pt",
  "tfevent_path": "./tfevent_train/local_finetune"
}
```

## 4. Run Fine-tuning

Use `luna16_finetune.py` with a pretrained model:

```bash
python3 luna16_finetune.py \
  -e ./config/environment_local_finetune.json \
  -c ./config/config_train_luna16_16g.json \
  -p ./trained_models/model_luna16_fold1.pt \
  --max-epochs 100 \
  --head-only-epochs 20 \
  --finetune-lr 1e-4
```

Recommended schedule:

```text
Stage 1: freeze backbone and train detection heads.
Stage 2: unfreeze full network and continue fine-tuning.
```

If the local dataset is very similar to LUNA16, you can skip the head-only stage:

```bash
python3 luna16_finetune.py \
  -e ./config/environment_local_finetune.json \
  -c ./config/config_train_luna16_16g.json \
  -p ./trained_models/model_luna16_fold1.pt \
  --max-epochs 80 \
  --head-only-epochs 0 \
  --finetune-lr 1e-4
```

## 5. Evaluation

Recommended experiment:

```text
1. Run the pretrained model on the local test set as baseline.
2. Fine-tune on local training and validation data.
3. Run the fine-tuned model on the same local test set.
4. Compare CPM, low-FP sensitivity, and average false positives per scan.
```

The validation set is for checkpoint selection. The test set is for final reporting only.

## 6. Annotation Notes

Make sure local annotations follow a clear policy:

```text
- same minimum nodule size threshold
- consistent benign/malignant inclusion rules
- complete annotation of visible target nodules
- same coordinate convention as the training config
```

Incomplete annotations can bias evaluation because true nodules predicted by the model may be counted as false positives.
