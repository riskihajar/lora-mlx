# Working Doc-to-LoRA Workflow

This repository now includes a practical document-to-adapter pipeline:

1. Convert a document into supervised memory QA examples.
2. Train a LoRA adapter on those examples.
3. Query the base model with that adapter and no source context.

This is not SakanaAI's instant hypernetwork implementation. SakanaAI Doc-to-LoRA uses a trained hypernetwork to modulate LoRA weights from a document immediately. The workflow here is a fully runnable MLX LoRA baseline that creates a real `.npz` adapter for each document.

## Build Dataset

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/build_doc_to_lora_dataset.py \
  --input examples/my_document.txt \
  --output-dir data/doc_to_lora/my_document \
  --title my_document
```

Output:

- `data/doc_to_lora/my_document/train.jsonl`
- `data/doc_to_lora/my_document/valid.jsonl`
- `data/doc_to_lora/my_document/test.jsonl`
- `data/doc_to_lora/my_document/manifest.json`

## Train Adapter

```bash
source ~/.zshrc && scripts/train_doc_to_lora_tinyllama.sh examples/my_document.txt my_document 300
```

Output:

- `outputs/adapters/doc_to_lora_my_document.npz`

## Query Adapter

```bash
source ~/.zshrc && scripts/query_doc_to_lora_tinyllama.sh my_document "Apa fakta utama dari dokumen ini?"
```

## Evaluate Adapter

```bash
source ~/.zshrc && PYTHONPATH=src python3 -m lora_mlx.export \
  --model mlx_model \
  --adapter-file outputs/adapters/doc_to_lora_my_document.npz \
  --data data/doc_to_lora/my_document/test.jsonl \
  --output outputs/predictions/doc_to_lora_my_document.jsonl \
  --lora-layers 4 \
  --max-new-tokens 160
```

## Interpretation

- Use this workflow when you want a working document-specific adapter today.
- Use SakanaAI's hypernetwork approach as the future target if the thesis needs instant internalization without per-document optimization.
- The right thesis framing is: this repo now has a runnable Doc-to-LoRA baseline; matching SakanaAI's full D2L requires adding a hypernetwork architecture and training objective, not just LoRA fine-tuning.
