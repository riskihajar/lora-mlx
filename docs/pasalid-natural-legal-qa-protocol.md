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

- `D` lebih faktual daripada `B`;
- `D` citation lebih baik daripada `B`;
- `D` tidak sekadar menaikkan F1 dengan copy source mentah;
- `D` menjaga copy metrics lebih rendah atau setidaknya terkendali.

## Perintah Pipeline

Generate dataset natural dengan LLM-assisted generator:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/build_pasalid_natural_legal_qa.py --use-llm --limit 300 --questions-per-doc 3
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

## Kriteria Minimum Agar Layak Jadi Hasil Tesis

Eksperimen natural legal QA layak menjadi hasil utama jika:

- dataset final minimal `300` train dan `100` test gabungan seen/unseen;
- `D` mengungguli `B` pada answer quality atau citation quality;
- `D` tidak memiliki copy-rate yang lebih buruk secara ekstrem;
- minimal `30-50` contoh `B` vs `D` direview manual/LLM untuk factual correctness, evidence support, naturalness, dan citation correctness.

Dengan protokol ini, kontribusi penelitian tidak lagi bergantung pada rule-based template atau copy-paste source, tetapi pada adaptasi perilaku model dalam menjawab legal QA berbasis evidence.
