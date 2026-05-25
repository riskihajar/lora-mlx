# Cloud validation runs

Tooling for launching Hugging Face Jobs that validate the upstream
[`SakanaAI/doc-to-lora`](https://github.com/SakanaAI/doc-to-lora) repository on
a cloud GPU.

This is **Path 2** in `docs/sakanai-doc-to-lora-mlx-port-plan.md`: run upstream
PyTorch as-is on a CUDA host so we have a behavioral ground truth for the
MLX port (Path 1). All the Mac/MPS patches under `/Users/riskihajar/github/doc-to-lora`
are not used here; cloud uses upstream untouched.

## Prerequisites

- HF Pro account (required for GPU flavors).
- Local `hf` CLI logged in:
  ```bash
  hf auth login
  ```
- Approved access to the gated `google/gemma-2-2b-it` model on Hugging Face.

## Validation: upstream Sakana smoke test

Runs `git clone https://github.com/SakanaAI/doc-to-lora.git` inside the job,
mirrors `install.sh` (without the SQuAD/PWC/DROP/ROPES data builds),
downloads the `gemma_demo` checkpoint from `SakanaAI/doc-to-lora`, internalizes
`data/sakana_wiki.txt`, and prints the generated answer before and after
internalization.

```bash
./run_validate_sakana.sh
```

Expected runtime: roughly 25 to 45 minutes, dominated by `uv sync`,
flash-attn/flashinfer install, and the checkpoint download. Cost on
`a10g-small` at `$1.05/hour`: about `$0.50 to $1`.

Override defaults via env vars:

```bash
HF_JOB_FLAVOR=t4-medium HF_JOB_TIMEOUT=3h ./run_validate_sakana.sh
HF_JOB_DETACH=1 ./run_validate_sakana.sh    # detach and print job id
```

## Watch logs and manage jobs

```bash
hf jobs ps                  # list running jobs
hf jobs logs <job_id>       # tail logs
hf jobs logs -f <job_id>    # follow
hf jobs cancel <job_id>     # stop early
hf jobs inspect <job_id>    # detailed status
```

The smoke job prints both the baseline answer and the internalized answer to
stdout. No persistent storage; results live in the job logs.

## Files

| File | Role |
| --- | --- |
| `validate_sakana_upstream.sh` | Entry point executed inside the job container |
| `run_validate_sakana.sh`     | Local trigger that wraps `hf jobs run` |

## Next phases (not implemented yet)

After upstream is confirmed working in the cloud:

1. **Pasal.id swap**: replace `data/sakana_wiki.txt` with Pasal.id documents,
   run a small batch (e.g. 30 seen + 30 unseen samples), persist outputs to a
   private HF dataset, then pull back to `outputs/predictions/`.
2. **A/B/C/D parity**: extend the smoke script to emit the four conditions
   used in `docs/pasalid-thesis-experiment-report.md` for direct comparison
   against MLX port results.

These phases will get their own scripts under `cloud/` once Path 2 is green.
