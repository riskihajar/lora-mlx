# Laporan Eksperimen Pasal.id

## Konteks

- Tujuan eksperimen ini adalah mengevaluasi apakah internalisasi berbasis adapter dapat memindahkan pengetahuan dokumen hukum ke parameter model open-weight, sehingga model dapat menjawab pertanyaan tanpa menerima dokumen sumber saat inferensi.
- Sumber data berasal dari API Pasal.id, tetapi yang digunakan dalam repo ini adalah artefak turunan lokal yang telah dibersihkan.
- Tiga kondisi utama yang dibandingkan adalah:
  - `A`: model dasar tanpa konteks dokumen
  - `B`: model dasar dengan konteks dokumen sumber
  - `C`: model dasar dengan adapter LoRA tanpa konteks dokumen
- Untuk kualitas jawaban, `F1` diperlakukan sebagai metrik utama.
- `EM` tetap dilaporkan sebagai metrik ketat tambahan, tetapi tidak dijadikan dasar utama pembuktian karena jawaban legal QA dapat berbeda bentuk permukaan meskipun substansinya benar.

## Pipeline Data Saat Ini

- raw cache Pasal.id: `data/pasalid_raw/`
- canonical document units: `data/pasalid/doc_units.jsonl`
- QA bank utama: `data/pasalid/qa_bank_full.jsonl`
- split eksperimen utama: `data/pasalid/experiment_split/`
- QA bank JSON `answer + source` yang lebih stabil: `data/pasalid/qa_bank_json_large.jsonl`
- split JSON yang lebih besar: `data/pasalid/json_large_split/`

Ringkasan split JSON yang paling stabil saat ini:

- total rows: `180`
- train rows: `96`
- valid rows: `15`
- test seen rows: `48`
- test unseen rows: `21`

## Metrik yang Dipakai

### Kualitas jawaban

- `Answer F1` sebagai metrik utama kualitas jawaban
- `Answer EM` sebagai metrik ketat tambahan

### Akuntabilitas jawaban

- `Citation EM`
- `Citation Component Score`

Catatan penting:

- Sampai tahap eksperimen saat ini, metrik citation masih sangat lemah pada hampir semua model dan format.
- Artinya, eksperimen lebih kuat dalam menunjukkan kualitas internalisasi isi jawaban daripada kualitas keterlacakan sumber.

## Hasil Utama yang Stabil

### TinyLlama pada format JSON `answer + source`

Evaluasi seen split pada `20` contoh memberikan hasil berikut:

| Kondisi | Answer EM | Answer F1 | Citation EM | Citation Component Score |
| --- | ---: | ---: | ---: | ---: |
| A | 0.0000 | 0.3151 | 0.0000 | 0.0000 |
| B | 0.0000 | 0.5078 | 0.0000 | 0.0000 |
| C | 0.0000 | 0.3439 | 0.0000 | 0.0000 |

Interpretasi:

- Urutan performa stabil adalah `B > C > A`.
- Kondisi `B` menunjukkan bahwa pemberian konteks dokumen secara eksplisit masih menjadi pendekatan terkuat.
- Kondisi `C` mengungguli `A`, sehingga ada sinyal bahwa adapter berhasil menginternalisasi sebagian pengetahuan dokumen untuk meningkatkan kualitas jawaban.
- Namun, peningkatan pada kualitas jawaban tidak diikuti oleh peningkatan kualitas attribution sumber.

### TinyLlama pada split unseen

Smoke test unseen split pada `10` contoh sebelumnya menunjukkan:

| Kondisi | EM | F1 |
| --- | ---: | ---: |
| A | 0.0000 | 0.2599 |
| B | 0.0000 | 0.2917 |
| C | 0.0000 | 0.2412 |

Interpretasi:

- Keuntungan konteks masih ada pada unseen split.
- Adapter tidak lagi mengungguli baseline no-context pada subset unseen ini.
- Ini menunjukkan bahwa sinyal internalisasi saat ini lebih kuat pada seen-document setting daripada generalisasi ke unseen-document setting.

## Perbandingan Antar Model

### Mistral q4

Pada setup JSON `answer + source`, seen split `10` contoh:

| Kondisi | Answer F1 | Citation Component Score |
| --- | ---: | ---: |
| A | 0.2283 | 0.0000 |
| B | 0.3898 | 0.0000 |
| C | 0.2622 | 0.0000 |

Catatan:

- `C` masih lebih baik daripada `A`, tetapi tetap jauh di bawah `B`.
- Mistral belum mengungguli TinyLlama pada setup Pasal.id saat ini.
- Perpanjangan training Mistral memang memberi sedikit kenaikan, tetapi belum cukup untuk mengubah ranking model.

### Qwen3

Pada setup JSON `answer + source`, seen split `20` contoh:

| Kondisi | Answer F1 | Citation Component Score |
| --- | ---: | ---: |
| A | 0.2041 | 0.0000 |
| B | 0.3773 | 0.0000 |
| C | 0.2400 | 0.0000 |

Catatan:

- `C` mengungguli `A`, tetapi masih tidak mendekati `B` sebaik TinyLlama.
- Qwen3 belum menggeser TinyLlama sebagai baseline terbaik dalam konfigurasi eksperimen Pasal.id saat ini.

### Ranking praktis saat ini

Dalam setup JSON `answer + source` yang paling stabil saat ini, ranking praktisnya adalah:

1. `TinyLlama`
2. `Qwen3`
3. `Mistral q4`

Ranking ini spesifik untuk:

- corpus Pasal.id-derived saat ini
- split eksperimen saat ini
- format jawaban yang sedang diuji
- checkpoint adapter yang saat ini tersedia

## Ablasi Format Jawaban

Beberapa format jawaban telah dicoba:

1. jawaban naratif bebas
2. format dua baris `Jawaban + Sumber`
3. JSON `answer + source`
4. source-components (`source_type`, `source_number`, `source_year`, `source_article`)

Temuan utamanya:

- format JSON `answer + source` adalah format yang paling layak dipertahankan untuk eksperimen lanjutan
- format source-components tidak memperbaiki traceability, sehingga lebih tepat diperlakukan sebagai hasil ablasi negatif
- memperketat format target jawaban membantu evaluasi, tetapi belum menyelesaikan masalah source adherence

## Temuan Utama

- `B` secara konsisten merupakan kondisi terkuat di semua model yang diuji.
- `C` dalam beberapa setup mampu mengungguli `A`, yang mendukung klaim bahwa adapter dapat menginternalisasi sebagian informasi dokumen.
- Kualitas jawaban dan kualitas keterlacakan sumber saat ini terpisah secara empiris: jawaban bisa membaik tanpa diikuti attribution yang baik.
- Memperbesar model tidak otomatis memperbaiki masalah traceability.

## Batas Klaim yang Aman

- Aman untuk menyatakan bahwa adapter-based internalization dapat meningkatkan kualitas jawaban dibanding kondisi no-context pada beberapa setup.
- Aman untuk menyatakan bahwa pendekatan context-based masih lebih kuat daripada adapter-only inference dalam eksperimen ini.
- Belum aman untuk menyatakan bahwa model sudah mampu memberi source attribution yang reliabel dan machine-checkable tanpa intervensi tambahan.

## Bottleneck Utama Saat Ini

- source traceability masih menjadi kelemahan utama di seluruh model dan format yang diuji
- sebagian sample masih cukup panjang dan berat untuk model kecil
- kualitas QA bank sudah cukup untuk baseline, tetapi masih dapat diperbaiki lebih lanjut jika eksperimen akan diperluas

## Arah Lanjutan yang Paling Rasional

1. pertahankan format JSON `answer + source` sebagai format utama eksperimen berikutnya
2. perlakukan source-components sebagai hasil ablasi, bukan jalur utama lanjutan
3. jika eksperimen diteruskan, arah paling bernilai adalah task khusus source attribution atau citation prediction, bukan sekadar memperbesar model lagi
