# ITaylor: Hierarchical Integration of Sequence and Molecular Surface Features for pHLA–TCR Recognition

ITaylor predicts pHLA–TCR recognition by combining paired sequence information with
surface features derived independently from monomeric pHLA and TCR structures.

> **Zenodo — Data and Model Assets:** [10.5281/zenodo.22063489](https://doi.org/10.5281/zenodo.22063489)  
> **Zenodo — Complete Software and Data:** [10.5281/zenodo.22065527](https://doi.org/10.5281/zenodo.22065527)

## Installation

The provided environment targets Linux with Python 3.12 and CUDA-enabled PyTorch:

```bash
conda env create -f environment.yml
conda activate ITaylor
```

The sequence predictor loads EvolutionaryScale ESM-C 600M through
`ESMC.from_pretrained("esmc_600m")`. The model must be available in the local cache or
downloadable on first use.

## Download model and data files

Large files are not stored in Git. Download the Zenodo archives and extract them into the
repository root while preserving the following paths:

```text
sequence_weight/neg_ratio_10/fold_{0..4}/best_model.pt
surface_weight/neg_ratio_10/neg_ratio_10/fold_{0..4}/best_model.pt
data/Database_stage1/09_FINAL_deduped_reindexed_cleaned.csv
data/Database_stage2/imfp/
Adaptive_correction/fold_calibration_seqonly/fold_{0..4}_val_predictions.csv
Adaptive_correction/fold_calibration_surfonly/PANDORA_pdb_output/
Adaptive_correction/fold_calibration_surfonly/tcr_only_tFold/
Adaptive_correction/fold_calibration_surfonly/output_with_pdb_files.csv
```

Use `--preflight-only` before inference to identify missing packages or files.

## Single-sample prediction

Use `ITaylor_predict.py` for one pHLA–TCR sample:

```bash
python ITaylor_predict.py \
  --epitope TLMSAMTNL \
  --hla_allele "HLA-A*02:01" \
  --hla_sequence "MATURE_HLA_SEQUENCE" \
  --tcra "TCR_ALPHA_VARIABLE_SEQUENCE" \
  --tcrb "TCR_BETA_VARIABLE_SEQUENCE" \
  --cdr3a CAVNNARLMF \
  --cdr3b CASSVAGSPEAFF \
  --pmhc_masif_dir example/9NMU_pHLA01 \
  --tcr_masif_dir example/9NMU_TCR01 \
  --seq_device cuda:0 \
  --surf_device cuda:1
```

The complete example sequences are available in `example/Example_detail.csv`. To
validate the same command without loading the models, append:

```text
--preflight-only
```

The program prints one score to standard output:

```text
ITaylor_score=0.81703830
```

### Required surface files

`--pmhc_masif_dir` must directly contain the pHLA `p1_*` files and
`--tcr_masif_dir` the TCR `p2_*` files:

```text
{p1|p2}_rho_wrt_center.npy
{p1|p2}_theta_wrt_center.npy
{p1|p2}_mask.npy
{p1|p2}_input_feat_charge.npy
{p1|p2}_input_feat_ddc.npy
{p1|p2}_input_feat_hbond.npy
{p1|p2}_input_feat_hphob.npy
{p1|p2}_input_feat_si.npy
```

Optional structure-quality arguments are `--phla_dope_per_res`, `--tcr_lddt`,
`--tcr_ptm`, and `--tcr_iptm`. Run `python ITaylor_predict.py --help` for all options.

## Batch testing

Use `ITaylor_test.py` for CSV files matching `TEST_*.csv`:

```bash
python ITaylor_test.py --preflight-only

python ITaylor_test.py \
  --seq_model_dir sequence_weight/neg_ratio_10 \
  --surf_model_dir surface_weight/neg_ratio_10 \
  --test_dir data/Database_stage1/test_outputs \
  --database_csv data/Database_stage1/09_FINAL_deduped_reindexed_cleaned.csv \
  --mhc_pseudo data/Database_stage1/MHC_psuedo.dat \
  --imfp_dir data/Database_stage2/imfp \
  --output_dir predictions/ITaylor_test \
  --seq_device cuda:0 \
  --surf_device cuda:1 \
  --batch_size 64
```

Each input CSV must contain:

```text
id,hla_allele,Epitope,hla_alpha123_mature,tcra_variable,tcrb_variable,id_tcr
```

The optional `label` column enables AUROC and AUPRC calculation. By default,
`ITaylor_test.py` writes all outputs to `predictions/ITaylor_test/`. Use
`--output_dir` to select a different directory. The output files are:

```text
ITaylor_test.log
<input>_meta_N5_predictions.csv
<input>_metrics.json
summary_metrics_N5.json
```

The prediction CSV contains the final `ITaylor_score` column. Metric JSON files are
only created when labels are present.

## Training

### Stage 1: sequence model

Script: `Train_stage1_logging.py`

First generate the positive-sequence ESM-C embeddings from
`data/Database_stage1/outputs_split/training_positive.csv`:

```bash
python ESM-process.py
```

`ESM-process.py` uses fixed input/output paths and writes the filtered embeddings and
metadata to `data/Database_stage1/esm-embedding/`.

Required files:

```text
data/Database_stage1/esm-embedding/training_positive_clear_peplen7_esm_embeddings_float32.pt
data/Database_stage1/esm-embedding/training_positive_metadata_clear_peplen7.pt
data/Database_stage1/outputs_split/training_negative_clear_peplen7_merged.csv
```

Train all five folds:

```bash
python Train_stage1_logging.py \
  --data_dir data/Database_stage1/esm-embedding \
  --neg_ratio 10 \
  --output_dir sequence_weight \
  --training_details_dir sequence_weight/running_details \
  --device cuda:0 \
  --max_epochs 50
```

Add `--fold 0`, ..., `--fold 4` to train only one fold. Models are saved under:

```text
sequence_weight/neg_ratio_10/fold_<N>/
```

### Stage 2: surface model

Script: `Train_stage2_surfonly_logging.py`

Required files:

```text
data/Database_stage1/outputs_split/training_positive_clear_peplen7.csv
data/Database_stage1/outputs_split/training_negative_clear_peplen7_v1.csv
data/Database_stage2/imfp/train_pmhc/pmhc_<ID>/p1_*.npy
data/Database_stage2/imfp/train_tcr/tcr_<ID>/p2_*.npy
```

Train one fold:

```bash
python Train_stage2_surfonly_logging.py \
  --imfp_dir data/Database_stage2/imfp \
  --pos_csv_path data/Database_stage1/outputs_split/training_positive_clear_peplen7.csv \
  --neg_csv_path data/Database_stage1/outputs_split/training_negative_clear_peplen7_v1.csv \
  --fold 0 \
  --gpu 0 \
  --neg_ratio 10 \
  --max_epochs 30 \
  --output_dir surface_weight/neg_ratio_10
```

Repeat the command for folds 1–4. Models are saved under:

```text
surface_weight/neg_ratio_10/neg_ratio_10/fold_<N>/
```

Use the configuration JSON stored with each released model for exact reproduction.

### Adaptive correction

XGBoost and AlphaNet training is documented in
[`Adaptive_correction/README.md`](Adaptive_correction/README.md).

## Predictor APIs

The lower-level predictors can be imported directly when only one model branch is
needed.

```python
from predict_stage1_logging import Stage1Predictor

predictor = Stage1Predictor(
    model_dir="sequence_weight/neg_ratio_10",
    device="cuda:0",
)
probability = predictor.predict_single(epitope, hla_sequence, tcra, tcrb)
```

```python
from predict_stage2_surfonly_logging import Stage2SurfOnlyPredictor

predictor = Stage2SurfOnlyPredictor(
    model_dir="surface_weight/neg_ratio_10",
    imfp_dir="data/Database_stage2/imfp",
    device="cuda:0",
)
probability = predictor.predict_single(pmhc_id, tcr_id)
```

These branch probabilities are not the final `ITaylor_score`; use
`ITaylor_predict.py` or `ITaylor_test.py` for the complete pipeline.

## License

This project is licensed under the Apache License 2.0.
See the [LICENSE](LICENSE) file for details.
