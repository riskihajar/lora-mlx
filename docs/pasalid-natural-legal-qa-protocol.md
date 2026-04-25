# Pasal.id Natural Legal QA Protocol

## Tujuan Eksperimen

Tujuan core eksperimen ini bukan membuat interface chat, tetapi merancang dan mengevaluasi konfigurasi LoRA untuk legal QA berbasis dokumen hukum Indonesia. Eksperimen diarahkan untuk melihat konfigurasi mana yang paling efektif agar model kecil dapat:

- memahami pertanyaan user-style yang tidak menyebut "dokumen ini";
- menjawab dengan bahasa Indonesia natural dan ringkas;
- tetap grounded pada dokumen sumber;
- menyebut citation yang benar;
- tidak hanya copy-paste pasal.

## Fokus Eksperimen

Fokus eksperimen untuk tesis:

> Membandingkan konfigurasi base model, prompt-only context, adapter-only inference, dan adapter-with-context pada task retrieval-grounded legal QA natural, untuk mengetahui dampak LoRA terhadap kualitas jawaban, keterlacakan sumber, tingkat copy-paste dari dokumen, dan efisiensi inferensi.

Eksperimen ini menggeser fokus dari sekadar adapter-only memorization ke evaluasi **context-use behavioral adaptation** yang lebih dekat dengan kebutuhan sistem legal QA berbasis retrieval.

## Dataset Target

Dataset baru dibuat oleh `scripts/build_pasalid_natural_legal_qa.py`.

Karakter target:

- pertanyaan user-style, misalnya "Kalau seseorang mengirim ancaman lewat media elektronik, apa konsekuensi hukumnya?";
- jawaban natural 1-3 kalimat;
- jawaban tidak menyalin mentah pasal;
- citation tetap terstruktur dalam JSON;
- sumber tetap dari `doc_units` agar ground truth dapat diaudit.

Mode generation:

- `--use-llm`: jalur utama untuk menghasilkan pertanyaan dan jawaban natural paraphrase;
- tanpa `--use-llm`: bootstrap heuristic untuk smoke test pipeline, bukan dataset final tesis.

Filter final:

- unit dokumen dengan marker noise OCR dibuang;
- unit laporan keuangan/report-like dibuang karena lebih cocok untuk table/numeric QA daripada legal QA normatif;
- respons LLM yang bukan JSON valid dilewati agar satu respons rusak tidak menggagalkan seluruh build;
- held-out law dipilih dari law dengan jumlah row cukup agar `test_unseen` tidak terlalu kecil.
- contoh targeted ditambahkan untuk pasal peralihan/repeal agar pola "masih berlaku sepanjang tidak bertentangan" dan "dicabut/dinyatakan tidak berlaku" lebih terwakili.

## Kondisi Eksperimen

| Kondisi | Makna |
| --- | --- |
| `A` | base model tanpa konteks |
| `B` | base model dengan dokumen sumber |
| `C` | LoRA tanpa konteks |
| `D` | LoRA dengan dokumen sumber |

Bukti eksperimen utama sebaiknya berasal dari perbandingan `B` vs `D`:

- Jika `D > B` pada factual/natural/citation dan copy-rate lebih rendah, hasil eksperimen menunjukkan bahwa LoRA memberi manfaat di atas prompt-only retrieval.
- Jika `C` rendah, hasil tersebut tetap informatif karena menunjukkan batas adapter-only inference pada legal QA yang membutuhkan akuntabilitas sumber.

## Metrik Utama

Evaluator baru `scripts/eval_pasalid_natural_metrics.py` menambahkan metrik yang relevan untuk menghindari kesimpulan yang keliru dari jawaban copy-paste:

| Metrik | Tujuan |
| --- | --- |
| `answer_f1` | overlap substansi dengan target natural |
| `citation_em` | citation lengkap benar |
| `citation_component_score` | kecocokan parsial tipe/nomor/tahun/pasal |
| `valid_json_rate` | kepatuhan output machine-readable |
| `avg_max_source_copy_run` | panjang maksimum frasa source yang disalin berurutan |
| `avg_source_4gram_copy_ratio` | proporsi 4-gram jawaban yang berasal dari source |
| `copy_run_gt_10_rate` | rasio jawaban yang terlalu extractive |

Manfaat konfigurasi LoRA harus dievaluasi sebagai kombinasi:

- `D` lebih faktual daripada `B`, atau setidaknya mendekati `B` pada held-out law;
- `D` citation lebih baik daripada `B`;
- `D` tidak sekadar menaikkan F1 dengan copy source mentah;
- `D` menjaga copy metrics lebih rendah atau setidaknya terkendali.

## Perintah Pipeline

Generate dataset natural dengan LLM-assisted generator:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/build_pasalid_natural_legal_qa.py --use-llm --limit 600 --questions-per-doc 5
```

Smoke test tanpa LLM:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/build_pasalid_natural_legal_qa.py --limit 80
```

Train adapter natural legal TinyLlama:

```bash
source ~/.zshrc && scripts/train_pasalid_natural_legal_tinyllama.sh
```

Evaluasi A/B/C/D:

```bash
source ~/.zshrc && scripts/eval_pasalid_natural_legal_tinyllama.sh seen
source ~/.zshrc && scripts/eval_pasalid_natural_legal_tinyllama.sh unseen
```

Review pairwise `B` vs `D`:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/review_pasalid_pair_with_llm.py --b-input outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_seen_B_base_with_context.jsonl --d-input outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_seen_D_adapter_with_context.jsonl --output outputs/reviews/pasalid_natural_legal/seen_B_vs_D_pairwise_review.jsonl --limit 30
source ~/.zshrc && PYTHONPATH=src python3 scripts/review_pasalid_pair_with_llm.py --b-input outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_unseen_B_base_with_context.jsonl --d-input outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_unseen_D_adapter_with_context.jsonl --output outputs/reviews/pasalid_natural_legal/unseen_B_vs_D_pairwise_review.jsonl --limit 30
```

Ringkas review pairwise:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/summarize_pasalid_pairwise_review.py --input outputs/reviews/pasalid_natural_legal/seen_B_vs_D_pairwise_review.jsonl --output outputs/reviews/pasalid_natural_legal/seen_B_vs_D_pairwise_summary.json
source ~/.zshrc && PYTHONPATH=src python3 scripts/summarize_pasalid_pairwise_review.py --input outputs/reviews/pasalid_natural_legal/unseen_B_vs_D_pairwise_review.jsonl --output outputs/reviews/pasalid_natural_legal/unseen_B_vs_D_pairwise_summary.json
```

Post-process constrained JSON/source untuk menguji dampak format constraint tanpa retraining:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/postprocess_pasalid_natural_predictions.py --input outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_seen_B_base_with_context.jsonl --output outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_seen_B_base_with_context_constrained.jsonl
source ~/.zshrc && PYTHONPATH=src python3 scripts/postprocess_pasalid_natural_predictions.py --input outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_seen_D_adapter_with_context.jsonl --output outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_seen_D_adapter_with_context_constrained.jsonl
source ~/.zshrc && PYTHONPATH=src python3 scripts/postprocess_pasalid_natural_predictions.py --input outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_unseen_B_base_with_context.jsonl --output outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_unseen_B_base_with_context_constrained.jsonl
source ~/.zshrc && PYTHONPATH=src python3 scripts/postprocess_pasalid_natural_predictions.py --input outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_unseen_D_adapter_with_context.jsonl --output outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_unseen_D_adapter_with_context_constrained.jsonl
```

## Hasil Final Saat Ini

Dataset final LLM-assisted filtered dengan targeted completeness:

| Item | Nilai |
| --- | ---: |
| total rows | 538 |
| train rows | 340 |
| valid rows | 38 |
| test seen rows | 122 |
| test unseen rows | 38 |
| laws | 17 |
| targeted completeness/transition rows | 154 |

Hasil otomatis TinyLlama natural legal:

| Split | B F1 | D F1 | D - B | D Citation EM | D Copy >10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen | 0.2707 | 0.3341 | +0.0634 | 0.5738 | 0.2459 |
| unseen | 0.2902 | 0.3432 | +0.0530 | 0.6579 | 0.1842 |

Review pairwise `30` contoh per split:

| Split | Overall B Wins | D Wins | Ties |
| --- | ---: | ---: | ---: |
| seen | 10 | 17 | 3 |
| unseen | 8 | 16 | 6 |

Benchmark efisiensi per-example `20` contoh per split, `--max-new-tokens 96`:

| Split | Kondisi | Avg prompt tokens | Latency avg | Latency p95 | Generated tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| seen | A | 40.2 | 4.6856 | 4.8243 | 20.4989 |
| seen | B | 224.6 | 4.7001 | 5.2113 | 15.2868 |
| seen | C | 40.2 | 9.7087 | 10.2435 | 7.9568 |
| seen | D | 224.6 | 7.1523 | 10.0381 | 9.9898 |
| unseen | A | 33.0 | 3.6250 | 4.7000 | 18.9930 |
| unseen | B | 260.4 | 4.2514 | 5.2415 | 16.4651 |
| unseen | C | 33.0 | 9.8435 | 10.5666 | 5.9633 |
| unseen | D | 260.4 | 7.6658 | 10.2440 | 9.2619 |

Interpretasi efisiensi: `B` adalah baseline retrieval-only paling efisien, `C` mengurangi prompt token tetapi paling lambat karena biaya adapter inference, dan `D` menukar latency tambahan dengan source discipline serta naturalness yang lebih baik.

Hasil constrained JSON/source:

| Split | Kondisi | Answer F1 | Answer Recall | Citation EM | Valid JSON | Prompt Echo |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| seen | B constrained | 0.2791 | 0.2790 | 1.0000 | 1.0000 | 0.0000 |
| seen | D constrained | 0.3256 | 0.2994 | 1.0000 | 1.0000 | 0.0082 |
| unseen | B constrained | 0.2936 | 0.2860 | 1.0000 | 1.0000 | 0.0000 |
| unseen | D constrained | 0.3379 | 0.3054 | 1.0000 | 1.0000 | 0.0000 |

Review pairwise constrained `30` contoh per split:

| Split | Overall B Wins | D Wins | Ties | Catatan |
| --- | ---: | ---: | ---: | --- |
| seen | 12 | 11 | 7 | Hampir tie setelah source/JSON dipaksa sama-sama rapi |
| unseen | 11 | 13 | 6 | `D` unggul tipis dan tetap lebih rendah copy-rate |

Interpretasi constrained: format/source constraint menyelesaikan JSON, citation, dan prompt echo, tetapi tidak otomatis menaikkan recall. Setelah targeted completeness, `D` tetap unggul F1 atas `B` pada constrained seen/unseen, sementara pairwise constrained menjadi lebih seimbang.

## Kriteria Minimum Agar Layak Jadi Hasil Tesis

Eksperimen natural legal QA layak menjadi hasil utama jika:

- dataset final minimal `300` train dan `100` test gabungan seen/unseen;
- `D` mengungguli atau mendekati `B` pada answer quality, dan mengungguli `B` pada citation/source discipline;
- `D` memiliki copy-rate yang lebih rendah atau setidaknya tidak lebih buruk secara ekstrem;
- minimal `30-50` contoh `B` vs `D` direview manual/LLM untuk factual correctness, evidence support, naturalness, dan citation correctness.

Dengan protokol ini, kontribusi penelitian tidak lagi bergantung pada rule-based template atau copy-paste source, tetapi pada adaptasi perilaku model dalam menjawab legal QA berbasis evidence.
