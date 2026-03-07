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
streamlit run dashboards/datasets_review.py
streamlit run dashboards/embedding_pairs_review.py
streamlit run dashboards/embeddings_browse.py
```
