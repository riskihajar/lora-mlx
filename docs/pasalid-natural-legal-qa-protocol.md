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

Dataset final LLM-assisted filtered dengan targeted slot-repair:

| Item | Nilai |
| --- | ---: |
| total rows | 541 |
| train rows | 341 |
| valid rows | 39 |
| test seen rows | 121 |
| test unseen rows | 40 |
| laws | 17 |
| targeted slot-repair rows | 79 |
| targeted completeness/transition rows | 135 |

Hasil otomatis TinyLlama natural legal:

| Split | B F1 | D F1 | D - B | D Citation EM | D Copy >10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen | 0.2564 | 0.2886 | +0.0322 | 0.4959 | 0.1322 |
| unseen | 0.3180 | 0.3101 | -0.0079 | 0.5750 | 0.1500 |

Review pairwise `30` contoh per split:

| Split | Overall B Wins | D Wins | Ties |
| --- | ---: | ---: | ---: |
| seen | 13 | 16 | 1 |
| unseen | 14 | 15 | 1 |

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
| seen | B constrained | 0.2592 | 0.2503 | 1.0000 | 1.0000 | 0.0083 |
| seen | D constrained | 0.2848 | 0.2557 | 1.0000 | 1.0000 | 0.0165 |
| unseen | B constrained | 0.3210 | 0.3023 | 1.0000 | 1.0000 | 0.0000 |
| unseen | D constrained | 0.3051 | 0.2789 | 1.0000 | 1.0000 | 0.0000 |

Review pairwise constrained `30` contoh per split belum direrun setelah slot-repair. Tabel berikut adalah run constrained sebelumnya:

| Split | Overall B Wins | D Wins | Ties | Catatan |
| --- | ---: | ---: | ---: | --- |
| seen | 12 | 11 | 7 | Hampir tie setelah source/JSON dipaksa sama-sama rapi |
| unseen | 11 | 13 | 6 | `D` unggul tipis dan tetap lebih rendah copy-rate |

Interpretasi constrained: format/source constraint menyelesaikan JSON dan citation, tetapi tidak otomatis menaikkan recall atau slot accuracy. Setelah targeted slot-repair, `D` tetap lebih baik pada constrained seen F1, tetapi lebih rendah dari `B` pada constrained unseen F1.

Audit failure terbaru dapat direplikasi dengan:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/summarize_pasalid_failure_audit.py --input outputs/reviews/pasalid_natural_legal/seen_B_vs_D_pairwise_review.jsonl --output outputs/reviews/pasalid_natural_legal/seen_D_failure_audit_summary.json --model-key D
source ~/.zshrc && PYTHONPATH=src python3 scripts/summarize_pasalid_failure_audit.py --input outputs/reviews/pasalid_natural_legal/unseen_B_vs_D_pairwise_review.jsonl --output outputs/reviews/pasalid_natural_legal/unseen_D_failure_audit_summary.json --model-key D
```

Ringkasan residual failure `D`:

| Split | Broad Failures | Strict Failures | Dominant Residual Pattern |
| --- | ---: | ---: | --- |
| seen | 27/30 | 21/30 | incomplete focus, source/format, unnatural/echo |
| unseen | 23/30 | 20/30 | incomplete focus, factual/evidence, entity/count/list |

Interpretasi audit: targeted slot-repair memperbaiki sebagian slot held-out, tetapi strict failure tetap tinggi; sisa bottleneck utama berada pada validasi isi jawaban untuk provinsi asal, Lembaran Negara, daftar wilayah, dan status repeal.

Evaluator factual slot ditambahkan untuk mengukur correctness fakta yang dapat diekstraksi tanpa LLM judge:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/eval_pasalid_natural_slots.py --predictions outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_seen_D_adapter_with_context.jsonl --output outputs/reviews/pasalid_natural_legal/seen_D_slot_eval.json
source ~/.zshrc && PYTHONPATH=src python3 scripts/eval_pasalid_natural_slots.py --predictions outputs/predictions/pasalid_natural_legal/tinyllama_natural_legal_unseen_D_adapter_with_context.jsonl --output outputs/reviews/pasalid_natural_legal/unseen_D_slot_eval.json
```

Ringkasan factual slot raw:

| Split | B Correct / Total | B Acc. | D Correct / Total | D Acc. |
| --- | ---: | ---: | ---: | ---: |
| seen | 25/44 | 0.5682 | 22/44 | 0.5000 |
| unseen | 10/21 | 0.4762 | 9/21 | 0.4286 |

Interpretasi slot: slot-repair menaikkan `D` unseen dari `0.1765` ke `0.4286`, tetapi `B` masih sedikit lebih tinggi dan `D` turun pada seen. Constraint JSON/source tidak mengubah slot accuracy, sehingga perbaikan berikutnya harus berupa checker/decoding berbasis isi, bukan sekadar data tambahan.

## Kriteria Minimum Agar Layak Jadi Hasil Tesis

Eksperimen natural legal QA layak menjadi hasil utama jika:

- dataset final minimal `300` train dan `100` test gabungan seen/unseen;
- `D` mengungguli atau mendekati `B` pada answer quality, dan mengungguli `B` pada citation/source discipline;
- `D` memiliki copy-rate yang lebih rendah atau setidaknya tidak lebih buruk secara ekstrem;
- minimal `30-50` contoh `B` vs `D` direview manual/LLM untuk factual correctness, evidence support, naturalness, dan citation correctness.

Dengan protokol ini, kontribusi penelitian tidak lagi bergantung pada rule-based template atau copy-paste source, tetapi pada adaptasi perilaku model dalam menjawab legal QA berbasis evidence.
