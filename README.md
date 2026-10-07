# SCL-MGEN

*Uncertainty-aware Multilingual AI-Generated Text Detection Using Contrastive Empowered Multi-Granular Evidential Network*.

This guide covers environment setup, obtaining the datasets from their official providers, preparing input files, training a SCL-MGEN, and evaluating it.

## 1. Training a model

Training starts from the pretrained `BAAI/bge-m3` encoder. The refinement, fusion, contrastive projection and classification modules are initialized, and the two-stage detector training runs from epoch 1.

The primary configuration uses 10 contrastive epochs followed by 10 supervised evidential epochs. BGE-M3 is frozen for the first five contrastive epochs, progressively unfrozen during the next five, and fully trainable in Stage 2.

## 2. Set up the environment

Open a terminal in this repository's root folder, where `train.py`, `config.yaml` and `requirements.txt` are located. Use Python 3.12, Git for the M4 download, and an NVIDIA GPU with a compatible driver for FP16 training protocol. The project used an A100 with 80 GB.

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is unavailable, invoke `.\.venv\Scripts\python.exe` instead of `python` in the commands below.

For Linux or Windows with a CUDA 12.1-compatible NVIDIA driver, install the PyTorch CUDA build first, then the remaining dependencies:

```bash
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt 
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

The CUDA check must report `True` for the default FP16 configuration. If your system requires another supported CUDA build, use the matching PyTorch 2.3.1 command from the [official installation archive](https://pytorch.org/get-started/previous-versions/#v231). CPU-only PyTorch cannot run the FP16 protocol.

Create the data folders:

```bash
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/raw', 'data/processed']]"
```

Internet access is needed for installation, dataset acquisition, and the initial download of [BGE-M3 weights and tokenizer](https://huggingface.co/BAAI/bge-m3). The training script retrieves the configured backbone automatically.

## 3. Obtain MULTITuDE from the official provider

Use the [official Zenodo record](https://zenodo.org/records/10013755), linked by the [dataset authors' repository](https://github.com/kinit-sk/mgt-detection-benchmark).

1. Open the Zenodo record and sign in.
2. Request access using an institutional research email, following the record's conditions.
3. After approval, download the released dataset and extract the CSV file locally.
4. Place the dataset CSV at `data/raw/multitude.csv`, or use its actual local path in the preparation command.
5. Record the downloaded version and checksum. Do not replace the linked release with an extension or newer version.

The provider restricts access to research use and prohibits resharing outside the approved request. Keep the dataset out of a public code repository and cite its paper. Approval is required; these instructions do not bypass access controls. See the [provider's access terms](https://zenodo.org/records/10013755).


Convert the CSV:

```bash
python preprocess.py --input data/raw/multitude.csv --output data/processed/multitude.jsonl
```

This produces `multitude.jsonl` and `multitude.audit.json`. The script rejects invalid labels, missing language/split metadata, duplicate IDs/texts, and known source leakage. If validation fails, inspect the reported issue against the original data rather than silently dropping records or changing splits.

Record the source checksum:

```bash
python -c "from pathlib import Path; import hashlib; p=Path('data/raw/multitude.csv'); print(hashlib.sha256(p.read_bytes()).hexdigest(), p)"
```

## 4. Obtain M4 from the official repository

Use the [official M4 repository](https://github.com/mbzuai-nlp/M4) and its [data directory](https://github.com/mbzuai-nlp/M4/tree/main/data):

```bash
git clone https://github.com/mbzuai-nlp/M4.git data/raw/M4
git -C data/raw/M4 rev-parse HEAD
```

You can also download the repository archive from GitHub's Code menu and extract it to `data/raw/M4`.

The [official data documentation](https://github.com/mbzuai-nlp/M4/blob/main/data/README.md) describes domain/generator JSONL files with fields such as `human_text`, `machine_text`, `model`, `source`, and `source_ID`. Inspect the selected files before conversion; filenames and available fields can vary. Human and machine texts can be extracted from paired fields in the same source file when present.

The converter requires explicitly partitioned input files and a manifest. Obtain the target experiment's subset and split manifest before claiming an exact M4 reproduction. If you construct a new partition, keep related source records together, record the selection and splitting procedure, and identify the run as an alternative protocol. A field named `split` in a raw file is not automatically used by this converter: `split` is assigned per manifest entry, so each selected input file must belong to one declared partition.

To prepare a verified selection:

1. Obtain or construct the partitioned source files with recorded provenance. Do not assign a whole unpartitioned raw file to both train and test.
2. Copy `../m4_manifest.example.json` to `../m4_manifest.json`.
3. Replace the example entries with every selected file, including both human and machine records in train and test.
4. For each entry, specify `path`, `language`, `domain`, `split`, `text_field`, `label`, `generator`, and actual `sha256`. Paths are relative to the manifest's directory; a source under this repository's data directory therefore normally starts with `../data/`.
5. Set `partition_provenance` to the verifiable selection/split source, or the complete procedure used for an alternative protocol.

Compute a checksum for each selected file, replacing the example path with your actual partitioned file:

```bash
python -c "from pathlib import Path; import hashlib; p=Path('data/raw/M4_partitioned/train/wikipedia_chatgpt.jsonl'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Example manifest structure below uses placeholder paths and checksums. Add the corresponding test files and all required languages/generators:

```json
{
  "partition_provenance": "Describe the actual subset and split provenance here",
  "files": [
    {
      "path": "../data/raw/M4_partitioned/train/wikipedia_chatgpt.jsonl",
      "language": "en",
      "domain": "wikipedia",
      "split": "train",
      "text_field": "human_text",
      "label": 0,
      "generator": "human",
      "sha256": "REPLACE_WITH_ACTUAL_SHA256"
    },
    {
      "path": "../data/raw/M4_partitioned/train/wikipedia_chatgpt.jsonl",
      "language": "en",
      "domain": "wikipedia",
      "split": "train",
      "text_field": "machine_text",
      "label": 1,
      "generator": "chatGPT",
      "sha256": "REPLACE_WITH_ACTUAL_SHA256"
    }
  ]
}
```

Run the converter once the manifest is complete:

```bash
python -m scripts.prepare_m4 --manifest configs/m4_manifest.json --output data/processed/m4.jsonl
```

It verifies source checksums, deduplicates repeated human records within partitions, preserves available source IDs, and audits the resulting records. Keep the generated and source manifest alongside experiment records. Follow the dataset providers' terms and cite [the M4 paper](https://aclanthology.org/2024.eacl-long.83/).

## 5. Training configuration


| Setting | Primary configuration |
|---|---|
| Backbone | BAAI/bge-m3 |
| Maximum input length | 8192 tokens |
| Granularities | Word, sentence, document |
| Refinement | Two transformer blocks per stream |
| Stage 1 | 10 epochs, batch 32 with 16 examples per class, LR 2e-5 |
| Stage 2 | 10 epochs, batch 16, accumulation 4, LR 1e-5 |
| Optimizer | AdamW, weight decay 0.01 |
| Contrastive temperature | 0.07 |
| Evidential loss | Printed squared-error plus Dirichlet KL, beta 0.1 |
| Composite loss coefficient | 1.0 |
| Precision / seed | FP16 / 42 |
| Evaluation | Five folds; validation-selected threshold, minimum coverage 90%; ECE with 15 bins |

The implementation also selects defaults for details the manuscript does not fully specify, including refinement/fusion dimensions, projection widths, segmentation, warm-up, progressive unfreezing. It saves resolved settings for each fold. Changing length, batch size, precision or architecture for a smaller GPU changes the experimental configuration.

## 6. Train from epoch 1

After data preparation, start one fold:

```bash
python train.py --config config.yaml --suite main --fold 0
```

The script initializes the encoder from BGE-M3, initializes the added modules, creates the fold partitions, performs both training stages, selects the best Stage 2 checkpoint using validation macro F1, selects the uncertainty threshold on validation data, and evaluates the model. 

Train all five folds:

```bash
python train.py --config config.yaml --suite main
```

For a complete, verified M4 manifest and prepared data:

```bash
python train.py --config configs/m4.yaml --suite main --fold 0
python train.py --config configs/m4.yaml --suite main
```

Fold indices are 0 through 4. Interrupted or mismatching runs also require a new output directory; mid-epoch resume is not implemented. Changing the parent output directory affects both dataset configurations unless overridden.

The main experiment is the `full` variant. Each fold writes files under `../<dataset>/full/fold_<index>/`, including:

- `config.json`, `splits.json`, and `environment.json`;
- `history.json` and `run.json`;
- `../best.pt`, the saved tokenizer, and backbone configuration;
- validation, internal-test, outer-test and official-test predictions and metrics.

These paths use the configured `output_dir`; adjust evaluation paths if you change it.

## 7. Evaluate a Model

Evaluate the official test partition using the validation-selected threshold:

The evaluator script writes `metrics.json` and `predictions.jsonl`, including accuracy, macro F1, 15-bin ECE and selective coverage/risk. It loads the trained model and local tokenizer/backbone configuration. The threshold's coverage constraint is enforced on validation data.

The standalone evaluator supports the dataset's `train` or `test` partition. Use the training-produced `metrics_outer_test.json` and `predictions_outer_test.jsonl` for outer-fold evaluation. Keep this separate from `official_test`. Report fold means and sample standard deviations for the chosen protocol.

## 8. Additional experiments and troubleshooting

Other training suites are `ablations`, `transfer`, `backbones`, `granularity`, `depth`, `sensitivity`, and `all`. 
```bash
python train.py --config config.yaml --suite all --dry-run
python train.py --config configs/m4.yaml --suite all --dry-run
```

They contain 56 MULTITuDE and 60 M4 configurations, respectively. Running `--suite all` without `--dry-run` trains those configurations across the configured folds and is much more expensive than the main experiment.

| Issue | Action |
|---|---|
| MULTITuDE download unavailable | Complete the official research-access request; the files are restricted. |
| M4 checksum or split validation fails | Correct the manifest using the original source files and verified partition provenance. |
| CUDA unavailable | Check the NVIDIA driver, active environment and installed PyTorch CUDA build. |
| CUDA out of memory | Use sufficient GPU memory for the primary configuration; record any configuration changes as a different experiment. |
| Existing incomplete or mismatching run | Start a fresh run. |

The minimal release does not include external baseline implementations, plotting scripts or unspecified learned-prior sensitivity. 
