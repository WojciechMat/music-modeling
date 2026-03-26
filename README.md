# Music Modeling

## Setup
```sh
pip install -r requirements.txt
```

## Build datasets (order matters)
```sh
python -m scripts.build_midi_dataset --num-proc 16
python -m scripts.build_midi_tokenized_jsonl ./midi_datasets/MidiDataset/latest --num-proc 16
python -m scripts.build_embedding_pairs_jsonl ./midi_datasets/MidiDataset/latest --num-proc 16
```
Each step writes to `midi_datasets/<Name>/latest`. Same config skips rebuild.

## Train

**Next-token prediction**
```sh
python -m training.next_token_prediction.train
```
Default data: `./midi_datasets/MidiTokenizedDataset/latest`.

**Note embedder** (ECHO, needs a pretrained next-token checkpoint)
```sh
python -m training.embedder.train pretrained_model_path=./checkpoints/<run>/final embedder.echo_repetitions=2
```

## Export embeddings
```sh
python -m scripts.export_embeddings_jsonl ./midi_datasets/EmbeddingPairsDataset/latest ./checkpoints/note-embedder/<run>/final
```
Writes `tmp/data/.../split_with_embeddings.jsonl`.

## Dashboards
```sh
python -m streamlit run dashboards/datasets_review.py
python -m streamlit run dashboards/embedding_pairs_review.py
python -m streamlit run dashboards/embeddings_browse.py
```


## Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
