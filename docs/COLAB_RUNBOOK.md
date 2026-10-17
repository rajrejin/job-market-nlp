# Colab runbook (optional)

Colab only speeds up the BERT fine-tuning. The project is complete and testable without it: the local run trains the distilled model with a reduced budget (see `configs/default.yaml`, `classifier.local_max_train`, one seed). These steps were prepared by the build but **not executed by it**; you run them yourself.

## What the Colab run adds

The full training set, the larger `deepset/gbert-base` model next to `distilbert/distilbert-base-german-cased`, and three seeds (0, 1, 2) for each.

## 1. Prepare the files locally

```
jobnlp package-colab
```

This writes:

| file | content | private? |
|---|---|---|
| `dist/jobnlp_code.zip` | the repository code, configs and this notebook | no personal data |
| `dist/colab_inputs/postings.parquet` | the processed corpus: third-party job posting text, redacted (no emails, phone numbers, URLs or contact names) | **yes: third-party data, keep private** |
| `dist/colab_inputs/split_manifest.json` | the fixed train, validation and test split | contains posting ids only |
| `dist/colab_inputs/expected.json` | checksums and the git commit the run must match | no personal data |

Never upload credentials, `.env` files or your CV.

## 2. Upload to your own Drive (private)

In Google Drive create the folder `MyDrive/jobnlp/` and upload into it:

```
MyDrive/jobnlp/
  jobnlp_code.zip
  colab_inputs/
    postings.parquet
    split_manifest.json
    expected.json
```

Do not share the folder or create a link to it. The posting texts come from public job feeds whose terms allow non-commercial use and no public display.

## 3. Run the notebook

1. Open https://colab.research.google.com, choose File > Upload notebook, and pick `notebooks/colab_finetune_bert.ipynb` from the project folder.
2. Runtime > Change runtime type > Hardware accelerator: **T4 GPU** > Save.
3. Runtime > Run all. Allow Drive access when asked (cell 2).
4. Cell 4 stops with an assertion error if a checksum does not match: run `jobnlp package-colab` again and upload the new files.
5. Expected duration on a T4: the distilled model takes minutes per seed; `gbert-base` takes longer. The free tier can end a session at any time and GPU availability varies. Checkpoints are saved to Drive every epoch, so after a disconnect reconnect and choose Run all again: finished runs are skipped and an interrupted run resumes from its last checkpoint.

## 4. Bring the models back

For each run, download the folder `MyDrive/jobnlp/outputs/<model>_seed<k>/model/` (it holds `model.safetensors`, `config.json`, the tokenizer files and `manifest.json`). Put each folder anywhere locally, for example `dist/colab_returned/gbert-base_seed0/model/`, then run:

```
jobnlp import-colab dist/colab_returned/gbert-base_seed0/model --name gbert_seed0
```

The import refuses a model whose manifest does not match the local git commit, config, corpus and split manifest, or whose weights are missing. Accepted models are copied to `models/classifier_bert_<name>/` and **re-evaluated locally** on the local test split; metrics computed inside Colab are never used. Results go to `results/classifier_<name>_eval.json` and are marked as Colab-trained in the README.
