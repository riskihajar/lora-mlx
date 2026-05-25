#!/usr/bin/env bash
# Entry point executed inside an HF Jobs container.
#
# Runs the upstream SakanaAI/doc-to-lora repository as-is on a CUDA GPU,
# downloads the gemma_demo checkpoint, internalizes data/sakana_wiki.txt,
# and prints generated answers before and after internalization.
#
# Designed to be passed via `bash -c "$(cat validate_sakana_upstream.sh)"`
# in `cloud/run_validate_sakana.sh`.

set -euo pipefail

echo "===== HF Jobs: Sakana D2L upstream validation ====="
echo "[info] start: $(date -u)"
echo "[info] hostname: $(hostname)"
nvidia-smi || true
python --version || true

# 0. Install minimal system deps (the pytorch:devel images ship without git/curl)
echo "[info] installing system deps (git, curl, ca-certificates) ..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends git curl ca-certificates >/dev/null

# 1. Clone upstream
cd /tmp
rm -rf doc-to-lora
echo "[info] cloning upstream repo ..."
git clone --depth 1 https://github.com/SakanaAI/doc-to-lora.git
cd doc-to-lora
echo "[info] upstream HEAD: $(git rev-parse HEAD)"

# 2. uv installer
echo "[info] installing uv ..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

# 3. Create Python 3.10 venv (matches upstream install.sh)
echo "[info] creating Python 3.10 venv ..."
uv venv --python 3.10 --seed

# 4. Install torch cu124 + project deps + flash-attn + flashinfer.
# This mirrors install.sh, but skips the SQuAD/PWC/DROP/ROPES dataset
# builds because the smoke test does not need them.
echo "[info] installing torch cu124 ..."
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --torch-backend=cu124

echo "[info] uv sync (full pyproject) ..."
uv sync

echo "[info] installing tokenizers/flash-attn/flashinfer ..."
uv pip install tokenizers==0.21.0
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6

# 5. Download Sakana D2L checkpoint (only the gemma_demo subdir, ~1.5 GB)
echo "[info] downloading Sakana D2L gemma_demo checkpoint ..."
HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download \
    SakanaAI/doc-to-lora \
    --local-dir trained_d2l \
    --include "gemma_demo/*"

# 6. Smoke inference: matches the upstream README example.
echo "[info] running smoke inference ..."
cat > smoke_inference.py <<'PYEOF'
import torch

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


CHECKPOINT = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
MAX_NEW_TOKENS = 256


def main() -> None:
    print(f"[smoke] loading checkpoint: {CHECKPOINT}")
    state_dict = torch.load(CHECKPOINT, weights_only=False)

    print("[smoke] building ModulatedPretrainedModel ...")
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,
        use_sequence_packing=False,
    )
    model.reset()

    tokenizer = get_tokenizer(model.base_model.name_or_path)

    doc = open("data/sakana_wiki.txt").read()
    print(f"[smoke] doc length (chars): {len(doc)}")

    chat = [{"role": "user", "content": "Tell me about Sakana AI."}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # A) baseline: no document, no adapter
    print("[smoke] generating BEFORE internalization (baseline) ...")
    outputs = model.generate(input_ids=chat_ids, max_new_tokens=MAX_NEW_TOKENS)
    print("\n===== A) baseline answer (no document, no adapter) =====")
    print(tokenizer.decode(outputs[0]))

    # C) internalized: hypernetwork-generated adapter, no doc in prompt
    print("\n[smoke] internalizing document ...")
    model.internalize(doc)

    print("[smoke] generating AFTER internalization ...")
    outputs = model.generate(input_ids=chat_ids, max_new_tokens=MAX_NEW_TOKENS)
    print("\n===== C) internalized answer (adapter, no doc in prompt) =====")
    print(tokenizer.decode(outputs[0]))

    print("\n[smoke] done.")


if __name__ == "__main__":
    main()
PYEOF

uv run python smoke_inference.py

echo "[info] finished: $(date -u)"
echo "===== success ====="
