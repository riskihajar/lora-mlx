# Laporan Eksperimen Pasal.id

## Konteks

- Tujuan eksperimen ini adalah mengevaluasi apakah internalisasi berbasis adapter dapat memindahkan pengetahuan dokumen hukum ke parameter model open-weight, sehingga model dapat menjawab pertanyaan tanpa menerima dokumen sumber saat inferensi.
- Sumber data berasal dari API Pasal.id, tetapi yang digunakan dalam repo ini adalah artefak turunan lokal yang telah dibersihkan.
- Kondisi utama yang dibandingkan adalah:
  - `A`: model dasar tanpa konteks dokumen
  - `B`: model dasar dengan konteks dokumen sumber
  - `C`: model dasar dengan adapter LoRA tanpa konteks dokumen
  - `D`: model dasar dengan adapter LoRA dan konteks dokumen sumber
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

### Percobaan split final dari QA bank naratif

Untuk memperbesar evaluasi, QA bank naratif `data/pasalid/qa_bank_full.jsonl` dikonversi ke format JSON `answer + source` dan dibagi menjadi split baru `data/pasalid/json_final_split/`.

Ringkasan split:

| Split | Rows |
| --- | ---: |
| train | 146 |
| valid | 15 |
| test_seen | 73 |
| test_unseen | 51 |

Adapter TinyLlama final dilatih dengan dua varian checkpoint: `1000` iterasi dan sanity-check `400` iterasi. Hasilnya:

| Checkpoint | Split | A F1 | B F1 | C F1 | C Citation Component |
| --- | --- | ---: | ---: | ---: | ---: |
| 1000 iter | seen | 0.2700 | 0.4091 | 0.2371 | 0.0479 |
| 1000 iter | unseen | 0.2752 | 0.4279 | 0.1743 | 0.0000 |
| 400 iter | seen | 0.2700 | 0.4091 | 0.2163 | 0.0171 |
| 400 iter | unseen | 0.2752 | 0.4279 | 0.1956 | 0.0000 |

Interpretasi:

- Split final hasil konversi naratif memperbesar evaluasi, tetapi tidak memperkuat klaim internalisasi.
- Pada split ini, kondisi `C` berada di bawah `A` pada seen dan unseen.
- Checkpoint `400` iterasi tidak memperbaiki pola, sehingga masalahnya bukan hanya overtraining di `1000` iterasi.
- Beberapa sample training masih memicu warning panjang token, dan QA hasil konversi memiliki kualitas target yang lebih tidak stabil daripada QA JSON-native.
- Karena itu, split ini lebih tepat diperlakukan sebagai **negative robustness check**, bukan sebagai pengganti setup utama JSON-large.

Implikasi untuk eksperimen final:

- Eksperimen utama tetap sebaiknya memakai QA yang sejak awal dihasilkan dalam format JSON `answer + source`, bukan konversi otomatis dari jawaban naratif.
- Jika ingin memperbesar dataset final, langkah yang lebih tepat adalah menghasilkan native JSON QA tambahan dari `doc_units`, lalu melakukan review kualitas, bukan mengonversi QA lama secara massal.

### Native JSON QA expanded dari `doc_units`

Sebagai tindak lanjut, dibuat QA bank native langsung dari `data/pasalid/doc_units.jsonl` tanpa mengonversi jawaban naratif lama. Builder baru `scripts/build_pasalid_native_json_qa.py` menghasilkan target jawaban dalam format JSON `answer + source` dengan template pertanyaan yang grounded ke teks pasal.

Ringkasan artefak:

| Artefak | Nilai |
| --- | ---: |
| input doc units usable | 137 |
| output QA rows | 439 |
| total laws | 17 |
| train rows | 182 |
| valid rows | 75 |
| test seen rows | 90 |
| test unseen rows | 92 |

File utama:

- QA bank: `data/pasalid/qa_bank_json_native_expanded.jsonl`
- split: `data/pasalid/json_native_expanded_split/`
- config TinyLlama: `configs/pasalid_experiment_native_expanded_tinyllama.yaml`
- train wrapper: `scripts/train_pasalid_experiment_native_expanded_tinyllama.sh`
- eval wrapper: `scripts/eval_pasalid_experiment_native_expanded_tinyllama.sh`

Interpretasi:

- Dataset ini lebih layak sebagai kandidat eksperimen final daripada split konversi naratif karena jawaban dan source dibuat langsung dari unit dokumen yang sama.
- Ukuran test seen/unseen sudah mendekati target minimal awal dan jauh lebih seimbang daripada JSON-large pilot.
- Kelemahannya adalah sebagian pertanyaan masih template-based, sehingga tetap perlu spot-check/manual review sebelum dipakai sebagai klaim final.

Training TinyLlama pada split native-expanded selesai sampai `1000` iterasi dengan adapter `outputs/adapters/adapters_pasalid_tinyllama_native_expanded.npz`. Selama training dan export masih muncul warning beberapa sequence melebihi `2048` token, sehingga hasil ini perlu dibaca bersama catatan bahwa pre-splitting/chunking masih perlu diperbaiki.

Hasil otomatis `A/B/C/D`:

| Split | A F1 | B F1 | C F1 | D F1 | C - A | D - B | C Citation Component | D Citation Component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seen (`90`) | 0.2533 | 0.3992 | 0.3084 | 0.5933 | +0.0551 | +0.1941 | 0.0444 | 0.2000 |
| unseen (`92`) | 0.2698 | 0.3241 | 0.2271 | 0.4386 | -0.0427 | +0.1145 | 0.0326 | 0.2228 |

Interpretasi hasil native-expanded:

- Kondisi `C` mengungguli `A` pada seen, tetapi turun di bawah `A` pada unseen; ini memperkuat batas klaim adapter-only internalization.
- Kondisi `D` mengungguli `B` dengan margin lebih besar daripada JSON-large pilot, baik pada seen maupun unseen.
- Citation component pada `D` naik ke sekitar `0.20`, jauh lebih baik daripada pilot tetapi masih belum cukup untuk klaim source traceability yang reliabel.
- Native-expanded menjadi kandidat setup utama baru untuk branch context-use adaptation, sedangkan adapter-only `C` tetap perlu diposisikan sebagai internalisasi parsial yang tidak stabil di held-out law.

### Kondisi D: adapter dengan konteks dokumen

Untuk menguji apakah LoRA lebih berguna sebagai **context-use adapter** daripada pengganti konteks dokumen, ditambahkan kondisi:

- `D`: model dasar + adapter LoRA + konteks dokumen sumber saat inferensi

Pada setup TinyLlama JSON-large, hasilnya:

| Split | A F1 | B F1 | C F1 | D F1 | D - B | D Citation Component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seen | 0.2555 | 0.4524 | 0.2770 | 0.4670 | +0.0145 | 0.0208 |
| unseen | 0.2408 | 0.3500 | 0.2639 | 0.3643 | +0.0143 | 0.0476 |

Interpretasi:

- Kondisi `D` mengungguli `B` pada seen dan unseen split.
- Gain F1 masih kecil, tetapi konsisten pada dua split.
- Citation component juga mulai bergerak pada `D`, walaupun masih terlalu lemah untuk klaim traceability penuh.
- Hasil ini mendukung framing tambahan bahwa LoRA dapat berfungsi sebagai adapter perilaku/domain yang membantu model memanfaatkan konteks dokumen, bukan hanya sebagai parametric memory tanpa konteks.

Implikasi untuk tesis:

- Klaim adapter-only internalization (`C`) tetap harus dibatasi.
- Namun, klaim context-use adaptation (`D > B`) menjadi arah yang lebih feasible dan lebih dekat dengan sistem legal QA berbasis retrieval/context.

Evaluasi lintas model memperlihatkan bahwa efek `D` bersifat **model-dependent**:

| Model | Split | B F1 | D F1 | D - B | Citation Component D | Catatan |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| TinyLlama | seen | 0.4524 | 0.4670 | +0.0145 | 0.0208 | `D > B` |
| TinyLlama | unseen | 0.3500 | 0.3643 | +0.0143 | 0.0476 | `D > B` |
| Qwen3 | seen | 0.3565 | 0.3027 | -0.0538 | 0.0000 | `D < B` |
| Qwen3 | unseen | 0.3902 | 0.3064 | -0.0838 | 0.0000 | `D < B` |
| Mistral q4 long | seen | 0.3528 | 0.3992 | +0.0463 | 0.0000 | `D > B` |
| Mistral q4 long | unseen | 0.3936 | 0.4497 | +0.0561 | 0.0000 | `D > B` |

Makna lintas model:

- TinyLlama dan Mistral q4 long mendukung hipotesis bahwa LoRA dapat bertindak sebagai context-use/domain adapter.
- Qwen3 menjadi counterexample penting: adapter tidak otomatis memperbaiki penggunaan konteks dan bahkan dapat mengganggu baseline context.
- Karena itu, klaim aman bukan “`D` selalu lebih baik daripada `B`”, melainkan “context-use adaptation punya sinyal positif pada sebagian model dan perlu divalidasi per arsitektur/checkpoint”.
- Citation tetap menjadi bottleneck karena kenaikan `D` terutama terjadi pada overlap isi jawaban, bukan keterlacakan sumber yang stabil.

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

- `B` secara konsisten lebih kuat daripada adapter-only `C` sebagai baseline konteks eksplisit.
- `C` dalam beberapa setup mampu mengungguli `A`, yang mendukung klaim bahwa adapter dapat menginternalisasi sebagian informasi dokumen.
- `D` memperlihatkan sinyal context-use adaptation pada TinyLlama dan Mistral q4 long, tetapi tidak pada Qwen3.
- Kualitas jawaban dan kualitas keterlacakan sumber saat ini terpisah secara empiris: jawaban bisa membaik tanpa diikuti attribution yang baik.
- Memperbesar model tidak otomatis memperbaiki masalah traceability.

## Temuan yang Stabil

- Kondisi `B` tetap menjadi upper bound praktis untuk adapter-only inference (`C`) di seluruh eksperimen yang sudah dijalankan.
- Kondisi `C` dalam beberapa setup memberi peningkatan atas `A`, sehingga ada dasar untuk menyatakan internalisasi parsial pada kualitas jawaban.
- Kondisi `D` layak dilaporkan sebagai cabang tambahan karena menunjukkan bahwa LoRA bisa membantu penggunaan konteks pada sebagian model.
- Format JSON `answer + source` adalah format eksperimen paling layak saat ini untuk menjaga kualitas jawaban sambil tetap memungkinkan evaluasi traceability.
- `TinyLlama` tetap menjadi baseline paling kuat pada konfigurasi Pasal.id yang sudah diuji.

## Metrik yang Masih Belum Bergerak

- `EM` hampir selalu tetap `0`.
- `Citation EM` hampir selalu tetap `0`.
- `Citation Component Score` masih lemah dan tidak stabil.

Makna dari pola ini adalah bahwa peningkatan performa saat ini lebih banyak terjadi pada kualitas isi jawaban daripada pada kedisiplinan model dalam menyebut sumber secara andal.

## Batas Klaim yang Aman

- Aman untuk menyatakan bahwa adapter-based internalization dapat meningkatkan kualitas jawaban dibanding kondisi no-context pada beberapa setup.
- Aman untuk menyatakan bahwa pendekatan context-based masih lebih kuat daripada adapter-only inference dalam eksperimen ini.
- Aman untuk menyatakan bahwa adapter dengan konteks (`D`) memberi sinyal positif pada sebagian model, tetapi efeknya belum universal.
- Belum aman untuk menyatakan bahwa model sudah mampu memberi source attribution yang reliabel dan machine-checkable tanpa intervensi tambahan.

## Bottleneck Utama Saat Ini

- source traceability masih menjadi kelemahan utama di seluruh model dan format yang diuji
- sebagian sample masih cukup panjang dan berat untuk model kecil
- kualitas QA bank sudah cukup untuk baseline, tetapi masih dapat diperbaiki lebih lanjut jika eksperimen akan diperluas

## Arah Lanjutan yang Paling Rasional

1. pertahankan format JSON `answer + source` sebagai format utama eksperimen berikutnya
2. perlakukan source-components sebagai hasil ablasi, bukan jalur utama lanjutan
3. jika eksperimen diteruskan, arah paling bernilai adalah task khusus source attribution atau citation prediction, bukan sekadar memperbesar model lagi

## Rekomendasi Setup Utama Tesis

Berdasarkan seluruh eksperimen yang sudah dijalankan, setup yang paling layak diposisikan sebagai **eksperimen utama tesis** saat ini adalah sebagai berikut.

### Eksperimen utama 1: QA tanpa dokumen sumber saat inferensi

- objective: menguji apakah adapter dapat menginternalisasi isi dokumen sehingga jawaban tanpa konteks tetap lebih baik daripada baseline no-context
- dataset format: JSON `answer + source`
- model utama: `TinyLlama`
- kondisi utama yang dilaporkan:
  - `A`: base tanpa konteks
  - `B`: base dengan konteks dokumen
  - `C`: base + adapter tanpa konteks
- metrik utama:
  - `F1`
- metrik tambahan:
  - `EM`
  - `Citation EM`
  - `Citation Component Score`
  - metrik efisiensi inferensi

Alasan pemilihan:

- TinyLlama adalah model yang paling konsisten memberi sinyal internalisasi pada branch QA utama.
- Format JSON `answer + source` adalah format yang paling stabil untuk menjaga kualitas jawaban sekaligus tetap memungkinkan evaluasi traceability.

### Eksperimen utama 2: source prediction

- objective: menguji apakah attribution sumber dapat diinternalisasi ketika dijadikan task eksplisit
- dataset format: source prediction JSON
- model utama: `Mistral q4`
- metrik utama:
  - `source_exact_match`
  - `source_component_score`

Alasan pemilihan:

- Mistral q4 memberi `source_component_score` tertinggi, sehingga paling cocok untuk mewakili branch attribution sumber.
- Branch ini melengkapi branch QA utama dengan menunjukkan bahwa source attribution lebih efektif dipelajari sebagai task tersendiri.

### Posisi eksperimen lain

- `Qwen3` diposisikan sebagai pembanding model yang lebih kuat, bukan sebagai kandidat utama saat ini.
- format dua-baris dan source-components diposisikan sebagai eksperimen ablasi atau eksperimen pendukung.
- review manual seed diposisikan sebagai bukti awal operasional untuk akuntabilitas jawaban, bukan hasil final.

## Pemisahan Bukti Utama dan Bukti Pendukung

### Bukti utama

- hasil A/B/C TinyLlama pada setup JSON `answer + source`
- hasil source prediction Mistral q4
- benchmark efisiensi inferensi dasar untuk A/B/C

### Bukti pendukung

- eksperimen Qwen3
- eksperimen Mistral pada QA utama
- eksperimen format dua-baris
- eksperimen source-components
- seed manual review

Dengan pemisahan ini, hasil tesis dapat dibangun di atas satu konfigurasi utama yang lebih bersih, sementara eksperimen lain tetap berguna untuk memperkuat pembahasan dan menjelaskan mengapa setup utama dipilih.

## Cabang Eksperimen Source Prediction

Karena metrik citation pada task QA utama tetap sangat lemah, source attribution kemudian dipisahkan menjadi task tersendiri.

### Desain task

- input: pertanyaan hukum
- target: JSON sumber terstruktur berisi:
  - `source_type`
  - `source_number`
  - `source_year`
  - `source_article`

### Hasil TinyLlama

| Metric | Nilai |
| --- | ---: |
| valid_json_rate | 0.6667 |
| source_exact_match | 0.2857 |
| source_component_score | 0.5357 |
| source_type_accuracy | 0.6667 |
| source_number_accuracy | 0.3333 |
| source_year_accuracy | 0.6667 |
| source_article_accuracy | 0.4762 |

Interpretasi:

- TinyLlama sudah mampu memprediksi sebagian citation dengan benar ketika source attribution dijadikan task utama.
- Meskipun `source_exact_match` masih terbatas, `source_component_score` menunjukkan bahwa sebagian komponen sumber sering berhasil diprediksi dengan benar.

### Hasil Mistral q4

| Metric | Nilai |
| --- | ---: |
| valid_json_rate | 0.9048 |
| source_exact_match | 0.3333 |
| source_component_score | 0.7262 |
| source_type_accuracy | 0.9048 |
| source_number_accuracy | 0.4286 |
| source_year_accuracy | 0.9048 |
| source_article_accuracy | 0.6667 |

Interpretasi:

- Pada task source prediction, Mistral q4 lebih kuat daripada TinyLlama.
- Ini menunjukkan bahwa model yang belum tentu paling kuat pada answer generation bisa menjadi model yang lebih baik untuk attribution sumber.

### Hasil Qwen3

| Metric | Nilai |
| --- | ---: |
| valid_json_rate | 0.8095 |
| source_exact_match | 0.3810 |
| source_component_score | 0.6548 |
| source_type_accuracy | 0.8095 |
| source_number_accuracy | 0.4762 |
| source_year_accuracy | 0.7619 |
| source_article_accuracy | 0.5714 |

Interpretasi:

- Pada task source prediction, Qwen3 menghasilkan `source_exact_match` tertinggi.
- Namun, `source_component_score` Qwen3 masih berada di bawah Mistral q4.
- Ini menunjukkan tradeoff yang menarik: Qwen3 lebih baik pada exact citation, sedangkan Mistral lebih baik pada kecocokan komponen secara keseluruhan.

### Makna untuk eksperimen

- Hasil ini menunjukkan bahwa answer generation dan source attribution sebaiknya tidak diperlakukan sebagai satu kemampuan tunggal.
- Pada eksperimen saat ini, QA utama lebih cocok untuk menilai internalisasi isi jawaban.
- Task source prediction lebih cocok untuk menilai internalisasi attribution sumber.
- Ranking model juga berbeda antar cabang eksperimen, sehingga model terbaik harus dipilih sesuai objective evaluasinya.

### Stress test source-prediction implicit

Untuk mengurangi risiko bahwa source-prediction hanya menyalin UU, tahun, atau pasal yang sudah muncul di prompt, dataset source attribution kemudian dipisahkan menjadi:

- `explicit`: pertanyaan boleh menyebut sumber hukum target
- `implicit`: pertanyaan tidak menyebut UU, tahun, atau pasal target secara eksplisit

Builder dataset juga menambahkan pertanyaan `implicit` berbasis isi jawaban agar evaluasi tidak terlalu kecil. Hasil awal pada `data/pasalid_source/implicit/test.jsonl` (`24` contoh) adalah:

| Model | valid_json_rate | source_exact_match | source_component_score | source_type_accuracy | source_number_accuracy | source_year_accuracy | source_article_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyLlama | 0.8333 | 0.0000 | 0.3438 | 0.8333 | 0.0000 | 0.5417 | 0.0000 |
| Mistral q4 | 1.0000 | 0.0000 | 0.5312 | 1.0000 | 0.0000 | 1.0000 | 0.1250 |
| Qwen3 | 0.5417 | 0.0000 | 0.2604 | 0.5417 | 0.0000 | 0.5000 | 0.0000 |

Interpretasi:

- Evaluasi `implicit` jauh lebih sulit daripada source-prediction awal yang banyak mengandung sumber eksplisit di prompt.
- Semua model gagal pada exact match ketika sumber target tidak muncul langsung di pertanyaan.
- Mistral q4 masih menjadi yang terkuat pada `source_component_score`, terutama karena konsisten menghasilkan JSON valid, source type, dan tahun.
- `source_number_accuracy` tetap `0`, sehingga kemampuan attribution belum cukup untuk klaim citation penuh.
- Hasil ini memperkuat batas klaim: source-prediction eksplisit berguna sebagai sanity check format, tetapi bukti attribution yang lebih kuat harus memakai setup `implicit`.

### Split source-prediction seen vs unseen

Setelah stress test awal, dataset source attribution diperbaiki lagi agar memiliki `test_seen` dan `test_unseen`, bukan hanya satu `test` berbasis held-out law. Ini penting karena source attribution implicit pada held-out law penuh menuntut model menebak nomor UU yang tidak pernah muncul di training.

Ringkasan split `data/pasalid_source/implicit/` setelah perbaikan:

| Split | Rows |
| --- | ---: |
| train | 125 |
| valid | 18 |
| test_seen | 71 |
| test_unseen | 24 |

Detector leakage menemukan `0` target-source mention pada train, valid, `test_seen`, dan `test_unseen`.

### Retraining Mistral q4 pada source-implicit

Adapter baru dilatih pada `data/pasalid_source/implicit/` dengan output `outputs/adapters/adapters_pasalid_source_implicit_mistral_q4.npz`. Hasilnya dibandingkan dengan adapter source Mistral lama pada split implicit yang sama:

| Adapter | Split | valid_json_rate | source_exact_match | source_component_score | source_number_accuracy | source_article_accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mistral source lama | implicit seen | 1.0000 | 0.0000 | 0.5493 | 0.0845 | 0.1268 |
| Mistral source-implicit baru | implicit seen | 0.8873 | 0.0141 | 0.5035 | 0.1408 | 0.1690 |
| Mistral source lama | implicit unseen | 1.0000 | 0.0000 | 0.5312 | 0.0000 | 0.1250 |
| Mistral source-implicit baru | implicit unseen | 0.9167 | 0.0000 | 0.5000 | 0.0000 | 0.1667 |

Interpretasi:

- Retraining khusus implicit belum memperbaiki `source_component_score` total.
- Adapter baru sedikit menaikkan `source_number_accuracy` dan `source_article_accuracy` pada seen split, tetapi menurunkan valid JSON rate, type accuracy, dan year accuracy.
- Pada unseen split, source number tetap `0`, sehingga generalisasi citation ke UU held-out belum tercapai.
- Hasil ini sebaiknya diperlakukan sebagai hasil negatif yang berguna: memperbesar/memurnikan implicit data saja belum cukup; perlu desain training atau objective yang lebih kuat jika attribution penuh menjadi target final.

### Implikasi tesis

- Jika citation metrics pada task QA utama tetap rendah, itu tidak lagi berarti source attribution gagal total.
- Sebaliknya, hasil source-prediction menunjukkan bahwa kemampuan attribution dapat muncul secara bermakna ketika task diformulasikan secara lebih langsung.

## Efisiensi Implementasi

Benchmark awal TinyLlama pada setup JSON `answer + source`, seen split, `10` contoh menghasilkan ringkasan berikut:

| Kondisi | Avg Prompt Token Proxy | Avg Latency (s) | p50 (s) | p95 (s) | Peak RSS Proxy (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 27.4 | 2.3434 | 2.3434 | 2.3434 | 1201733632 |
| B | 110.2 | 2.3609 | 2.3609 | 2.3609 | 2662760448 |
| C | 27.4 | 3.9444 | 3.9444 | 3.9444 | 2652045312 |

Interpretasi:

- Kondisi `C` berhasil menurunkan kebutuhan konteks inferensi secara nyata dibanding `B`, karena panjang prompt kembali mendekati `A`.
- Namun, pengurangan konteks tidak otomatis membuat inferensi lebih ringan pada latency maupun memory proxy.
- Dalam benchmark awal ini, `C` justru lebih lambat daripada `A` dan `B`, serta memiliki peak RSS proxy yang mendekati `B`.
- Artinya, manfaat utama `C` saat ini lebih jelas pada pengurangan kebutuhan konteks daripada efisiensi runtime murni.

## Review Manual Awal

Sebagai langkah awal untuk mengisi metrik akuntabilitas jawaban, dibuat seed manual review kecil pada beberapa contoh TinyLlama.

Ringkasan seed manual review (`6` baris):

| Metric | Nilai |
| --- | ---: |
| factual_correctness_avg | 0.6667 |
| evidence_support_avg | 0.6667 |
| source_traceability_avg | 0.0000 |
| evidence_support_rate | 0.3333 |
| unsupported_answer_rate | 0.6667 |
| factual_nonzero_rate | 0.3333 |
| source_traceability_rate | 0.0000 |

Interpretasi:

- Seed review ini masih terlalu kecil untuk dijadikan hasil akhir, tetapi sudah berguna sebagai bukti bahwa metrik manual dapat dihitung secara operasional.
- Nilai `unsupported_answer_rate` yang tinggi konsisten dengan temuan bahwa banyak jawaban masih belum cukup terikat ke evidence.
- `source_traceability_rate` yang tetap nol juga sejalan dengan lemahnya citation metrics pada task QA utama.
- Dengan demikian, review manual memperkuat kesimpulan dari metrik otomatis, bukan bertentangan dengannya.

### Seed manual review yang diperluas (`20` baris)

Untuk memberi sinyal yang sedikit lebih stabil, seed manual review kemudian diperluas menjadi `20` contoh pada TinyLlama kondisi `A` di seen split.

| Metric | Nilai |
| --- | ---: |
| factual_correctness_avg | 0.2000 |
| evidence_support_avg | 0.2000 |
| source_traceability_avg | 0.0000 |
| evidence_support_rate | 0.2000 |
| unsupported_answer_rate | 0.8000 |
| factual_nonzero_rate | 0.2000 |
| source_traceability_rate | 0.0000 |

Interpretasi:

- Hasil ini memperkuat pola dari seed review kecil sebelumnya.
- `unsupported_answer_rate` yang mencapai `0.8000` menunjukkan bahwa tanpa konteks dokumen, baseline TinyLlama masih sering menghasilkan jawaban yang tidak cukup didukung evidence.
- `source_traceability_rate` yang tetap `0.0000` juga selaras dengan hasil citation metrics otomatis di branch QA utama.
- Dengan demikian, aspek akuntabilitas jawaban sekarang tidak lagi kosong: walaupun masih berbasis sample manual awal, nilainya sudah bisa dilaporkan secara operasional.

## Review Semi-Otomatis Berbantu LLM

Untuk menghindari beban review manual penuh pada data yang lebih besar, dilakukan juga review semi-otomatis berbasis LLM sebagai alat pengelompokan awal.

### TinyLlama A/B/C pada JSON `answer + source`, seen split (`20` contoh per kondisi)

| Kondisi | factual_correctness_avg | evidence_support_avg | evidence_support_rate | unsupported_answer_rate | source_missing_rate | factually_wrong_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.0500 | 0.0500 | 0.0500 | 0.5500 | 0.8500 | 0.8000 |
| B | 0.9500 | 0.9000 | 0.7000 | 0.2500 | 1.0000 | 0.2500 |
| C | 0.1000 | 0.1000 | 0.0500 | 0.7500 | 0.9500 | 0.7500 |

Interpretasi:

- Hasil review semi-otomatis memperkuat pola utama bahwa kondisi `B` paling kuat tidak hanya pada F1, tetapi juga pada konsistensi faktual dan keterdukungan evidence.
- Kondisi `A` dan `C` sama-sama masih lemah dari sisi akuntabilitas jawaban, meskipun `C` pada beberapa metrik otomatis sempat menunjukkan gain atas `A`.
- `source_missing_rate` yang sangat tinggi pada ketiga kondisi menegaskan bahwa masalah utama tetap berada pada disiplin attribution sumber.
- Dengan demikian, hasil review semi-otomatis tidak membatalkan metrik otomatis, tetapi justru memperkuat interpretasi bahwa branch QA utama belum cukup untuk menghasilkan jawaban yang benar sekaligus akuntabel.

### Review semi-otomatis B vs D (`10` contoh per model/split)

Untuk memeriksa apakah kenaikan otomatis pada kondisi `D` hanya artefak overlap teks atau juga tampak pada penilaian kualitatif, dilakukan review berbantu LLM pada pasangan `B` dan `D` untuk subset awal `10` contoh per model/split.

| Model | Split | Δ factual D-B | Δ evidence D-B | Δ source D-B | Factual win B/D/tie | Evidence win B/D/tie | Source win B/D/tie |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| TinyLlama | seen | +0.20 | +0.20 | +0.50 | 2/3/5 | 2/3/5 | 1/4/5 |
| TinyLlama | unseen | -0.10 | -0.10 | +0.30 | 3/3/4 | 3/3/4 | 1/3/6 |
| Qwen3 | seen | -0.40 | -0.30 | +0.40 | 3/0/7 | 3/1/6 | 0/3/7 |
| Qwen3 | unseen | +0.10 | +0.10 | +0.10 | 3/4/3 | 3/4/3 | 2/3/5 |
| Mistral q4 long | seen | +0.40 | +0.30 | +0.20 | 1/4/5 | 2/4/4 | 4/3/3 |
| Mistral q4 long | unseen | +0.10 | +0.10 | +0.20 | 2/3/5 | 2/3/5 | 3/4/3 |

Interpretasi:

- Review awal mendukung sinyal `D` pada TinyLlama seen dan Mistral q4 long, terutama pada factual/evidence score.
- TinyLlama unseen lebih campuran: automatic F1 `D > B`, tetapi factual/evidence review tipis memihak `B` atau tie; ini perlu review lebih besar.
- Qwen3 tetap menjadi counterexample: seen split menunjukkan `D` lebih buruk secara factual/evidence, konsisten dengan automatic F1 `D < B`.
- Source traceability sering membaik pada `D`, tetapi skor absolut masih rendah dan tidak cukup untuk klaim citation reliability.
- Karena subset hanya `10` contoh per model/split dan berbantu LLM, hasil ini adalah **triangulasi awal**, bukan pengganti review manual final.

## Sintesis Akhir Antar Cabang

### Ringkasan cabang QA utama

- target utama: kualitas jawaban tanpa dokumen sumber saat inferensi
- hasil paling stabil: `B > C > A`
- model baseline terbaik saat ini: `TinyLlama`
- bottleneck utama: source traceability

### Ringkasan cabang source prediction

- target utama: prediksi sumber hukum secara eksplisit
- semua model menghasilkan metrik yang tidak lagi nol
- `Mistral q4` terbaik pada `source_component_score`
- `Qwen3` terbaik pada `source_exact_match`

### Kesimpulan lintas cabang

- kemampuan menghasilkan jawaban dan kemampuan memberi attribution sumber adalah dua kemampuan yang berbeda.
- eksperimen saat ini mendukung klaim bahwa internalisasi isi jawaban dapat terjadi tanpa konteks penuh.
- eksperimen saat ini juga menunjukkan bahwa internalisasi attribution sumber lebih efektif jika diformulasikan sebagai task khusus.
- dengan demikian, jika tujuan sistem akhir menuntut jawaban yang akuntabel, maka pendekatan dua-cabang atau dua-tahap lebih masuk akal daripada memaksa satu output generatif tunggal untuk menyelesaikan semuanya sekaligus.

## Eksperimen Source Prediction

Karena citation metrics pada task QA utama tetap rendah, task source attribution kemudian dipisahkan menjadi task tersendiri.

### Desain task

- input: pertanyaan hukum
- target: JSON sumber terstruktur berisi:
  - `source_type`
  - `source_number`
  - `source_year`
  - `source_article`

### Baseline TinyLlama

Dataset source-prediction dibangun dari QA bank JSON yang lebih besar dan dievaluasi dengan evaluator khusus source prediction.

Hasil baseline TinyLlama:

| Metric | Nilai |
| --- | ---: |
| valid_json_rate | 0.6667 |
| source_exact_match | 0.2857 |
| source_component_score | 0.5357 |
| source_type_accuracy | 0.6667 |
| source_number_accuracy | 0.3333 |
| source_year_accuracy | 0.6667 |
| source_article_accuracy | 0.4762 |

Interpretasi:

- Ketika source attribution dijadikan task tersendiri, metrik sumber akhirnya bergerak secara nyata.
- `source_exact_match` memang belum tinggi, tetapi sudah cukup untuk menunjukkan bahwa model mampu memprediksi sebagian citation dengan benar.
- `source_component_score` yang berada di atas `0.5` menunjukkan bahwa prediksi sumber parsial cukup sering benar, terutama pada jenis aturan dan tahun.
- Ini memperkuat pemisahan dua kemampuan: kualitas jawaban dapat dipelajari dalam task QA, sedangkan kualitas attribution sumber lebih efektif dipelajari sebagai task khusus.

Implikasi untuk narasi tesis:

- task QA utama lebih tepat dipakai untuk menilai internalisasi isi jawaban
- task source prediction lebih tepat dipakai untuk menilai internalisasi attribution sumber
- dengan demikian, citation metrics yang sebelumnya nol pada QA utama tidak lagi menjadi dead end, tetapi dialihkan ke branch evaluasi yang lebih tepat sasaran
