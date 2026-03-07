# Music Modeling

## Setup
```sh
pip install -r requirements.txt
```
All dependencies are pinned with `==` in `requirements.txt`.

Run all commands below from the **project root** (so `python -m scripts.*` and `python -m training.*` resolve correctly).

## Build datasets (order matters)
```sh
# 1) Aggregate MIDI from HuggingFace -> midi_datasets/MidiDataset/<hash>/ and midi_datasets/MidiDataset/latest
python -m scripts.build_midi_dataset --num-proc 4

# 2) Tokenize (use latest so no copy-paste)
python -m scripts.build_midi_tokenized_jsonl ./midi_datasets/MidiDataset/latest --num-proc 8

# 3) Embedding pairs (for embedder training)
python -m scripts.build_embedding_pairs_jsonl ./midi_datasets/MidiDataset/latest --num-proc 8
```
Each build updates a `latest` symlink so you can always pass `.../latest`. Same config skips rebuild.

## Train

Two training modes:

**Next-token prediction** (pretrain decoder on MIDI token sequences):
```sh
python -m training.next_token_prediction.train
```
Uses `data.dataset_path=./midi_datasets/MidiTokenizedDataset/latest` by default. Override: `data.dataset_path=./midi_datasets/MidiTokenizedDataset/<hash>`. Streaming: `data.streaming=true training.max_steps=50000`.

**Note embedder** (ECHO: decoder with repetitions, from a pretrained next-token checkpoint):
```sh
python -m training.embedder.train pretrained_model_path=./checkpoints/musical-transformer/<run> embedder.echo_repetitions=2
```
Config: `training/configs/embedder.yaml`. Required: `pretrained_model_path`. Hyperparameters: `embedder.echo_repetitions` (one value per run). Saves checkpoints in HuggingFace style (`best_eval_loss_model/`, `final/` with model weights and `tokenizer.json`).

## Export embeddings (test split)
Create embeddings for all test samples (notes_first) with an embedder checkpoint; writes `tmp/data/<dataset_hash>/<model_name>/split_with_embeddings.jsonl`.
```sh
python -m scripts.export_embeddings_jsonl ./midi_datasets/EmbeddingPairsDataset/latest ./checkpoints/note-embedder/<run>/final
```
Optional: `--split test`, `--output-root tmp/data`, `--max-seq-length 512`, `--batch-size 32`.

## Dashboards
```sh
streamlit run dashboards/datasets_review.py
streamlit run dashboards/embedding_pairs_review.py
streamlit run dashboards/embeddings_browse.py
```
Use path `./midi_datasets/MidiDataset/latest` or `./midi_datasets/MidiTokenizedDataset/latest` in the sidebar. For embeddings browse, use path to `split_with_embeddings.jsonl` (e.g. `tmp/data/<dataset_hash>/<model_name>/split_with_embeddings.jsonl`).
