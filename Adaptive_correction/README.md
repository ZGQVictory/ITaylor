# Adaptive correction

This directory contains the scripts and trained assets used to fit ITaylor's XGBoost
gate and AlphaNet correction model.

## Install the environment

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate ITaylor
```

## Train XGBoost and AlphaNet

Main script:

```text
Adaptive_correction/train_meta_xgboost_scaled_new_newsim_mlpalpha2_beta1.py
```

Required input:

```text
Adaptive_correction/calibration_meta_data/meta_aggregated_dataset.csv
```

Run:

```bash
python Adaptive_correction/train_meta_xgboost_scaled_new_newsim_mlpalpha2_beta1.py
```

The script uses fixed paths and does not accept command-line arguments. It removes
rows with missing required values, creates a stratified 80/20 train/validation split,
trains XGBoost, selects its ten most important features, and then trains AlphaNet.

### Training settings

| Component                   | Setting                                         |
| --------------------------- | ----------------------------------------------- |
| Data split                  | 80% train / 20% validation, stratified, seed 42 |
| XGBoost estimators          | 500                                             |
| XGBoost maximum depth       | 6                                               |
| XGBoost learning rate       | 0.05                                            |
| XGBoost row/column sampling | 0.8 / 0.8                                       |
| XGBoost early stopping      | 30 rounds                                       |
| AlphaNet                    | `Linear(12, 32) -> Tanh -> Linear(32, 1)`       |
| AlphaNet optimizer          | Adam, learning rate 0.01                        |
| AlphaNet epochs             | 1000, full-batch                                |
| AlphaNet selection          | Lowest validation BCE                           |


## Outputs

All outputs are written to `Adaptive_correction/calibration_meta_data/`:

| File                                       | Purpose                              |
| ------------------------------------------ | ------------------------------------ |
| `xgboost_meta_scaled_new_newsim.json`      | Trained XGBoost model                |
| `delta_hat_zscore_params_new_newsim.csv`   | XGBoost-output normalization values  |
| `alpha_top10_feat_names_beta1.csv`         | Selected feature names and order     |
| `alpha_top10_feat_mean_beta1.csv`          | AlphaNet feature means               |
| `alpha_top10_feat_std_beta1.csv`           | AlphaNet feature standard deviations |
| `alpha_net_mlpalpha2_beta1.pt`             | Best AlphaNet state dictionary       |
| `meta_mlpalpha2_beta1_params.csv`          | Validation summary                   |
| `meta_mlpalpha2_beta1_training_curves.png` | Training diagnostics                 |
| `meta_mlpalpha2_beta1_val_predictions.csv` | Per-sample validation predictions    |

The JSON, AlphaNet state, feature-name file, and normalization CSVs are required by
`ITaylor_predict.py` and `ITaylor_test.py`.

## Rebuild out-of-fold predictions

### Sequence branch

```bash
python Adaptive_correction/fold_validation_prediction_seqonly_newneg.py \
  --model_root sequence_weight/neg_ratio_10 \
  --pos_csv_path data/Database_stage1/outputs_split/training_positive_clear_peplen7.csv \
  --out_dir Adaptive_correction/fold_calibration_seqonly \
  --device cuda:0 \
  --folds all
```

Outputs:

```text
Adaptive_correction/fold_calibration_seqonly/fold_{0..4}_val_predictions.csv
```

### Surface branch

```bash
python Adaptive_correction/fold_validation_prediction_surfonly.py \
  --model_root surface_weight/neg_ratio10/neg_ratio_10 \
  --pos_csv_path data/Database_stage1/outputs_split/training_positive_clear_peplen7.csv \
  --out_dir Adaptive_correction/fold_calibration_surfonly \
  --device cuda:0 \
  --folds all
```

Outputs:

```text
Adaptive_correction/fold_calibration_surfonly/fold_{0..4}_val_predictions.csv
```

Both prediction scripts create `dataset_cache_*.npz` files. These caches are
regenerable and should not be committed. For parallel surface jobs, create the cache
once with `--prepare_cache_only`, then run separate `--folds 0`, ..., `--folds 4`
commands.

## Build the meta-feature table

`build_meta_features.py` requires the prepared intermediate table, out-of-fold
predictions, source database, PANDORA structures, and tFold structures. Ensure that the
following directories/files have been restored from Zenodo:

```text
Adaptive_correction/calibration_meta_data/meta_learning_data.csv
Adaptive_correction/fold_calibration_seqonly/fold_{0..4}_val_predictions.csv
Adaptive_correction/fold_calibration_surfonly/fold_{0..4}_val_predictions.csv
Adaptive_correction/fold_calibration_surfonly/PANDORA_pdb_output/
Adaptive_correction/fold_calibration_surfonly/tcr_only_tFold/
Adaptive_correction/fold_calibration_surfonly/output_with_pdb_files.csv
Adaptive_correction/fold_calibration_surfonly/negative_phla_pandora_with_ids.csv
data/Database_stage1/09_FINAL_deduped_reindexed_cleaned.csv
data/Database_stage1/MHC_psuedo.dat
```

Run:

```bash
python Adaptive_correction/build_meta_features.py
```

Output:

```text
Adaptive_correction/calibration_meta_data/meta_aggregated_dataset.csv
```

## Reproducibility notes

- Preserve the feature names and order stored with the trained models.
- Use the same XGBoost version for training and inference where possible.
- Keep `meta_aggregated_dataset.csv` and validation prediction tables in Zenodo rather
  than Git because they are large generated datasets.
