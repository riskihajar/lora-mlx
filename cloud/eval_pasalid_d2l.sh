#!/usr/bin/env bash
# HF Jobs entry point: evaluate upstream Sakana D2L on Pasal.id documents.
#
# Phases:
#   0. install minimal system deps + uv
#   1. clone upstream and install (mirrors install.sh, no dataset builds)
#   2. download SakanaAI/doc-to-lora gemma_demo checkpoint
#   3. download our private input dataset (riskihajar/pasalid-d2l-eval-input)
#   4. run inference: for each unique source_doc, internalize once and answer
#      every question that targets it; record latencies and outputs
#   5. push results to riskihajar/pasalid-d2l-eval-output
#
# Tunables via env vars (set in run_eval_pasalid.sh):
#   PASALID_INPUT_REPO   default: riskihajar/pasalid-d2l-eval-input
#   PASALID_OUTPUT_REPO  default: riskihajar/pasalid-d2l-eval-output
#   PASALID_SPLIT        default: test_seen,test_unseen (comma separated)
#   PASALID_MAX_NEW      default: 256
#   PASALID_RUN_TAG      default: $(date +%Y%m%d_%H%M%S)

set -euo pipefail

INPUT_REPO="${PASALID_INPUT_REPO:-riskihajar/pasalid-d2l-eval-input}"
OUTPUT_REPO="${PASALID_OUTPUT_REPO:-riskihajar/pasalid-d2l-eval-output}"
SPLITS="${PASALID_SPLIT:-test_seen,test_unseen}"
MAX_NEW="${PASALID_MAX_NEW:-256}"
RUN_TAG="${PASALID_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

echo "===== HF Jobs: Pasal.id Sakana D2L eval ====="
echo "[info] start: $(date -u)"
echo "[info] hostname: $(hostname)"
echo "[info] input_repo  = $INPUT_REPO"
echo "[info] output_repo = $OUTPUT_REPO"
echo "[info] splits      = $SPLITS"
echo "[info] max_new     = $MAX_NEW"
echo "[info] run_tag     = $RUN_TAG"
nvidia-smi || true
python --version || true

# 0. system deps
echo "[info] installing system deps (git, curl, ca-certificates) ..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends git curl ca-certificates >/dev/null

# 1. clone upstream
cd /tmp
rm -rf doc-to-lora
echo "[info] cloning upstream repo ..."
git clone --depth 1 https://github.com/SakanaAI/doc-to-lora.git
cd doc-to-lora
echo "[info] upstream HEAD: $(git rev-parse HEAD)"

# 2. uv installer + venv + deps (mirrors install.sh, skip dataset builds)
echo "[info] installing uv ..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

echo "[info] creating Python 3.10 venv ..."
uv venv --python 3.10 --seed

echo "[info] installing torch cu124 ..."
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --torch-backend=cu124

echo "[info] uv sync (full pyproject) ..."
uv sync

echo "[info] installing tokenizers/flash-attn/flashinfer ..."
uv pip install tokenizers==0.21.0
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6

# 3. download checkpoint and input dataset
echo "[info] downloading Sakana D2L gemma_demo checkpoint ..."
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download \
    SakanaAI/doc-to-lora \
    --local-dir trained_d2l \
    --include "gemma_demo/*"

echo "[info] downloading input dataset $INPUT_REPO ..."
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download \
    "$INPUT_REPO" \
    --repo-type dataset \
    --local-dir pasalid_input

ls -la pasalid_input/

# 4. eval script
echo "[info] writing eval driver ..."
cat > eval_pasalid.py <<'PYEOF'
"""Run upstream Sakana D2L on the local Pasal.id eval split.

Reads ``pasalid_input/test_<split>.jsonl`` and ``pasalid_input/docs_<split>.jsonl``
for each requested split, then for each unique source_doc:

  1. ``model.reset()``
  2. ``model.internalize(doc)``
  3. for each ``qa_id`` linked to that doc, generate an answer with the
     question alone (no doc in the prompt) and record latency

Writes ``pasalid_output/<run_tag>/predictions_<split>.jsonl`` and
``pasalid_output/<run_tag>/summary.json``. The wrapping bash script uploads
those to the output repo.
"""

import json
import os
import time
from pathlib import Path

import torch

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


CHECKPOINT = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
INPUT_DIR = Path("pasalid_input")
OUTPUT_BASE = Path("pasalid_output")
SPLITS = os.environ.get("PASALID_SPLIT", "test_seen,test_unseen").split(",")
MAX_NEW = int(os.environ.get("PASALID_MAX_NEW", "256"))
RUN_TAG = os.environ.get("PASALID_RUN_TAG", time.strftime("%Y%m%d_%H%M%S"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fp:
        return [json.loads(line) for line in fp]


def main() -> None:
    out_dir = OUTPUT_BASE / RUN_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] run_tag: {RUN_TAG}")
    print(f"[eval] output dir: {out_dir}")
    print(f"[eval] splits: {SPLITS}")
    print(f"[eval] max_new_tokens: {MAX_NEW}")

    print(f"[eval] loading checkpoint: {CHECKPOINT}")
    state_dict = torch.load(CHECKPOINT, weights_only=False)

    print("[eval] building ModulatedPretrainedModel ...")
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,
        use_sequence_packing=False,
    )
    model.reset()
    tokenizer = get_tokenizer(model.base_model.name_or_path)

    summary = {
        "run_tag": RUN_TAG,
        "checkpoint": CHECKPOINT,
        "max_new_tokens": MAX_NEW,
        "splits": {},
    }

    for split in SPLITS:
        split = split.strip()
        if not split:
            continue
        suffix = split.split("_", 1)[1]  # "seen" / "unseen"
        qa_path = INPUT_DIR / f"{split}.jsonl"
        doc_path = INPUT_DIR / f"docs_{suffix}.jsonl"
        if not qa_path.exists() or not doc_path.exists():
            print(f"[eval] missing split files for {split}; skipping")
            continue

        qa_rows = _read_jsonl(qa_path)
        doc_rows = _read_jsonl(doc_path)
        qa_by_id = {r["qa_id"]: r for r in qa_rows}
        print(f"[eval] split={split} qa={len(qa_rows)} docs={len(doc_rows)}")

        preds_path = out_dir / f"predictions_{split}.jsonl"
        n_pred = 0
        t_split = time.time()
        with preds_path.open("w") as fout:
            for doc_idx, doc_row in enumerate(doc_rows, 1):
                doc_id = doc_row["doc_id"]
                doc_text = doc_row["doc"]
                qa_ids = doc_row["qa_ids"]

                t0 = time.time()
                model.reset()
                model.internalize(doc_text)
                t_internalize = time.time() - t0

                print(
                    f"[eval] {split} doc {doc_idx}/{len(doc_rows)} {doc_id} "
                    f"len={len(doc_text)}c internalize={t_internalize:.2f}s "
                    f"qa={len(qa_ids)}"
                )

                for qa_id in qa_ids:
                    qa = qa_by_id[qa_id]
                    chat = [{"role": "user", "content": qa["question"]}]
                    chat_ids = tokenizer.apply_chat_template(
                        chat,
                        add_special_tokens=False,
                        return_attention_mask=False,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)

                    t1 = time.time()
                    outputs = model.generate(
                        input_ids=chat_ids, max_new_tokens=MAX_NEW
                    )
                    t_gen = time.time() - t1
                    text = tokenizer.decode(outputs[0])

                    record = {
                        "qa_id": qa_id,
                        "doc_id": doc_id,
                        "law_id": qa["law_id"],
                        "article_number": qa["article_number"],
                        "source_reference": qa["source_reference"],
                        "question": qa["question"],
                        "expected_answer": qa["expected_answer"],
                        "generated": text,
                        "latency_internalize_s": t_internalize,
                        "latency_generate_s": t_gen,
                        "max_new_tokens": MAX_NEW,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_pred += 1

        summary["splits"][split] = {
            "qa_predictions": n_pred,
            "docs": len(doc_rows),
            "wall_time_s": time.time() - t_split,
            "predictions_file": preds_path.name,
        }
        print(
            f"[eval] split={split} done: {n_pred} preds in "
            f"{summary['splits'][split]['wall_time_s']:.1f}s"
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[eval] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
PYEOF

# Pass tunables to python via env (already exported above through the parent
# shell; re-export so uv-run inherits them deterministically)
export PASALID_SPLIT="$SPLITS"
export PASALID_MAX_NEW="$MAX_NEW"
export PASALID_RUN_TAG="$RUN_TAG"

uv run python eval_pasalid.py

# 5. push outputs back to a private dataset
echo "[info] uploading outputs to $OUTPUT_REPO under run/$RUN_TAG ..."
uv run python - <<PYEOF
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="$OUTPUT_REPO", repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(
    folder_path="pasalid_output/$RUN_TAG",
    path_in_repo="run/$RUN_TAG",
    repo_id="$OUTPUT_REPO",
    repo_type="dataset",
)
print("[info] upload done.")
PYEOF

echo "[info] finished: $(date -u)"
echo "===== success ====="
