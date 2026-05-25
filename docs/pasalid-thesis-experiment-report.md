# Laporan Eksperimen Pasal.id

## Konteks

- Tujuan eksperimen ini adalah mengevaluasi dua peran adapter LoRA pada legal QA: sebagai adapter-only internalization (`C`) yang mencoba menjawab tanpa dokumen sumber saat inferensi, dan sebagai context-use adapter (`D`) yang membantu model memakai dokumen sumber secara lebih efektif.
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
- QA bank JSON `answer + source` pilot: `data/pasalid/qa_bank_json_large.jsonl`
- split JSON pilot: `data/pasalid/json_large_split/`
- QA bank native-expanded clean kandidat final: `data/pasalid/qa_bank_json_native_expanded_clean.jsonl`
- split native-expanded clean kandidat final: `data/pasalid/json_native_expanded_clean_split/`
- QA bank natural legal LLM-assisted: `data/pasalid/qa_bank_natural_legal.jsonl`
- split natural legal LLM-assisted: `data/pasalid/natural_legal_split/`

Ringkasan split native-expanded clean yang paling layak menjadi kandidat final saat ini:

- total rows: `432`
- train rows: `180`
- valid rows: `75`
- test seen rows: `89`
- test unseen rows: `88`
- warning sequence >2048 token: `0` pada train/valid/test dengan konteks

Catatan status split:

- `json_large_split/` tetap berguna sebagai pilot yang stabil untuk A/B/C dan lintas model.
- `json_native_expanded_clean_split/` lebih kuat sebagai kandidat final TinyLlama karena ukurannya lebih besar, split seen/unseen lebih seimbang, dan tidak memicu warning panjang token.

## Metrik yang Dipakai

### Kualitas jawaban

- `Answer F1` sebagai metrik utama kualitas jawaban
- `Answer EM` sebagai metrik ketat tambahan

### Akuntabilitas jawaban

- `Citation EM`
- `Citation Component Score`

Catatan penting:

- Pada A/B/C, metrik citation masih sangat lemah pada hampir semua model dan format.
- Pada clean split, `D` mulai menaikkan `Citation Component Score`, tetapi belum cukup untuk klaim source attribution yang reliabel.
- Artinya, eksperimen lebih kuat dalam menunjukkan peningkatan kualitas jawaban dan penggunaan konteks daripada keterlacakan sumber yang machine-checkable.

## Ringkasan Terbaru

Pada tahap terbaru, hasil paling penting berasal dari TinyLlama pada clean native-expanded split:

| Split | A F1 | B F1 | C F1 | D F1 | C - A | D - B | D Citation Component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seen (`89`) | 0.2618 | 0.4125 | 0.2617 | 0.6518 | -0.0001 | +0.2393 | 0.3146 |
| unseen (`88`) | 0.2720 | 0.3388 | 0.2210 | 0.4414 | -0.0510 | +0.1025 | 0.3494 |

Interpretasi ringkas:

- `D` adalah cabang paling kuat pada setup clean: LoRA paling berguna ketika dipakai bersama konteks dokumen.
- `C` tidak stabil sebagai adapter-only memory; pada clean split, `C` tidak mengungguli `A`.
- JSON-large tetap penting sebagai pilot historis, tetapi bukan lagi ringkasan hasil utama terbaru.

Tambahan natural legal QA LLM-assisted filtered menunjukkan tradeoff yang lebih realistis daripada template clean:

| Split | A F1 | B F1 | C F1 | D F1 | C - A | D - B | D Citation EM | D Citation Component | D Copy >10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seen (`119`) | 0.1828 | 0.2720 | 0.2065 | 0.3098 | +0.0237 | +0.0378 | 0.6639 | 0.6870 | 0.1681 |
| unseen (`39`) | 0.2302 | 0.3361 | 0.2381 | 0.3244 | +0.0078 | -0.0117 | 0.6410 | 0.6859 | 0.3333 |

Makna tambahan natural legal QA:

- Pada seen split, `D` masih mengungguli `B` pada answer F1 sekaligus jauh lebih baik pada citation dan valid JSON.
- Pada unseen split, `D` masih sedikit di bawah `B` pada answer F1 tetapi jauh lebih baik pada citation/format JSON; ini menunjukkan tradeoff antara answer overlap dan source discipline.
- `C` tetap tidak cukup sebagai adapter-only inference; nilainya hanya mendekati `A` dan jauh di bawah `B/D`.
- Copy-rate berbasis source masih perlu dikendalikan: `D` lebih rendah daripada `B` pada `copy_run_gt_10_rate`, tetapi tetap ada sekitar `17-33%` output yang terlalu extractive.
- Review pairwise LLM pada `30` seen dan `30` unseen menunjukkan `D` lebih kuat pada source traceability dan naturalness; setelah targeted transition augmentation, `D` menang overall pada seen dan unseen.

## Path 2: Sakana D2L Upstream Baseline

Selain pendekatan ordinary LoRA dan generated-LoRA MLX yang sudah dievaluasi di atas, tesis ini juga menjalankan **upstream `SakanaAI/doc-to-lora`** sebagai pembanding eksternal. Repo upstream `git@github.com:SakanaAI/doc-to-lora.git` dijalankan apa adanya di Hugging Face Jobs (CUDA), dengan data Pasal.id dari split `json_native_expanded_clean_split` (split yang sama dengan eksperimen MLX di atas).

Detail eksekusi:

- Job ID: `6a13ad63f17429a271eebf25` (full 177 row), tag `full_177`
- Flavor HF Jobs: `a10g-small` (provisioned A100-80GB)
- Checkpoint: `SakanaAI/doc-to-lora/gemma_demo/checkpoint-80000`
- Base model: `google/gemma-2-2b-it`
- LoRA target: `down_proj`, rank `8`
- Inference protocol: untuk setiap dokumen unik (`source_doc`), `model.reset()` lalu `model.internalize(doc)` sekali, lalu generate semua pertanyaan yang menargetkan dokumen tersebut tanpa konteks dokumen di prompt.
- Wall clock: ~22 menit total (`433s` test_seen + `348s` test_unseen + setup)
- Estimasi biaya: ~`$0.40`
- Output predictions: `outputs/predictions/pasalid_d2l_cloud/full_177/`
- Tooling: `cloud/build_pasalid_d2l_eval.py`, `cloud/eval_pasalid_d2l.sh`, `cloud/score_pasalid_d2l.py`.

Hasil pada split native-expanded clean:

| Split | Rows | Docs | Answer F1 | Answer EM | Source overlap | Internalize median (s) | Generate median (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_seen | 89 | 67 | 0.3231 | 0.0337 | 0.3740 | 1.00 | 2.53 |
| test_unseen | 88 | 26 | 0.2920 | 0.0114 | 0.3149 | 1.01 | 2.48 |

Perbandingan langsung pada split clean yang sama (TinyLlama dari eksperimen MLX di atas vs Sakana D2L upstream):

| Split | A (no doc, no adapter) | B (with doc, no adapter) | C TinyLlama (LoRA, no doc) | D TinyLlama (LoRA + doc) | Sakana D2L upstream (no doc) |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen | 0.2618 | 0.4125 | 0.2617 | 0.6518 | 0.3231 |
| unseen | 0.2720 | 0.3388 | 0.2210 | 0.4414 | 0.2920 |

Interpretasi:

- **Sakana D2L mengungguli ordinary-LoRA `C` pada kondisi yang sama** (no doc, hanya adapter): `+0.06` F1 pada seen dan `+0.07` F1 pada unseen. Ini sinyal bahwa pendekatan hypernetwork-generated adapter memang lebih kuat daripada fine-tuning LoRA per-document tradisional untuk internalisasi parametrik.
- **Sakana D2L masih jauh di bawah `B` dan `D`** yang menyertakan dokumen sumber di prompt. Konteks eksplisit tetap pendekatan terkuat untuk kualitas jawaban; D2L belum menggantikan retrieval, baru menjadi pengganti yang lebih baik untuk skenario tanpa konteks.
- **Source/citation tetap lemah**. `Source overlap` di rentang `0.31-0.37` karena adapter Sakana tidak dilatih dengan format JSON `answer + source` Pasal.id. Output Bahasa Indonesia natural, tetapi tidak terstruktur dan jarang menyebut nomor UU/pasal yang konsisten dengan referensi gold.
- **EM hampir nol** wajar: adapter Sakana dilatih pada SQuAD/PWC/DROP/ROPES Bahasa Inggris, sehingga tidak akan match exact ke target JSON. F1 token-overlap lebih informatif untuk kondisi ini.
- **Generalisasi lintas bahasa nyata**. Walaupun training data adapter berbahasa Inggris, output Pasal.id konsisten Bahasa Indonesia, mengangkat fakta dari dokumen yang baru di-internalize, dan strukturnya rapi. Beberapa halu masih muncul (misal nomor UU yang dibuat-buat), tetapi mayoritas jawaban faithfully mengangkat isi pasal.
- **Latency rendah**. Internalize ~1 detik per dokumen pendek (median panjang dokumen <500 karakter), generate ~2.5 detik. Cost dominasi setup (download checkpoint, install upstream stack), bukan inference.

Implikasi untuk tesis:

- Sakana D2L upstream berfungsi sebagai **ground-truth pembanding** untuk port MLX hypernetwork: angka F1 di atas adalah benchmark eksternal yang harus dikejar oleh implementasi MLX-mu pada split yang sama.
- Klaim tesis bisa diperkuat: ordinary-LoRA gagal sebagai memory adapter (`C ≈ A`), tetapi hypernetwork-based D2L jelas mengangkat F1 di atas baseline `A` dan ordinary `C`. Ini memvalidasi arah riset hypernetwork untuk Pasal.id.
- Gap source traceability adalah area kontribusi: D2L upstream tidak men-target citation Indonesia, sehingga port MLX yang dilatih dengan format JSON Pasal.id punya peluang menutup gap ini.

## Hasil Utama Historis dan Ablasi

### TinyLlama pada format JSON `answer + source`

Evaluasi seen split pada `20` contoh memberikan hasil berikut:

| Kondisi | Answer EM | Answer F1 | Citation EM | Citation Component Score |
| --- | ---: | ---: | ---: | ---: |
| A | 0.0000 | 0.3151 | 0.0000 | 0.0000 |
| B | 0.0000 | 0.5078 | 0.0000 | 0.0000 |
| C | 0.0000 | 0.3439 | 0.0000 | 0.0000 |

Interpretasi:

- Pada pilot JSON-large ini, urutan performa stabil adalah `B > C > A`.
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

Review berbantu LLM pada subset `20` contoh per split memberi triangulasi tambahan:

| Split | Δ factual D-B | Δ evidence D-B | Δ source D-B | Factual win B/D/tie | Evidence win B/D/tie | Source win B/D/tie |
| --- | ---: | ---: | ---: | --- | --- | --- |
| seen | +0.25 | +0.35 | +0.30 | 4/8/8 | 4/9/7 | 4/7/9 |
| unseen | -0.10 | 0.00 | +0.45 | 6/4/10 | 5/5/10 | 3/8/9 |

Makna review native-expanded:

- Pada seen split, review mendukung kenaikan `D` tidak hanya pada F1, tetapi juga factual correctness, evidence support, dan source traceability.
- Pada unseen split, `D` terutama memperbaiki source traceability; factual/evidence belum jelas membaik walaupun F1 otomatis naik.
- Dengan demikian, `D` layak menjadi cabang utama context-use adaptation, tetapi klaim generalisasi substantif pada unseen tetap perlu manual review lebih besar.

Analisis panjang token menunjukkan sumber warning >2048 token hanya berasal dari sedikit sample panjang:

| Split file | Rows | Max tokens | Over 2048 |
| --- | ---: | ---: | ---: |
| train | 182 | 2902 | 1 |
| valid | 75 | 630 | 0 |
| test_seen_with_context | 90 | 1594 | 0 |
| test_unseen_with_context | 92 | 8870 | 4 |

Untuk memperbaiki ini tanpa mengubah pipeline utama, builder native-expanded kini mendukung `--max-source-chars`. Dengan `--max-source-chars 3000`, dibuat split clean lokal:

| Clean split | Rows | Max tokens | Over 2048 |
| --- | ---: | ---: | ---: |
| train | 180 | 1756 | 0 |
| valid | 75 | 630 | 0 |
| test_seen_with_context | 89 | 1076 | 0 |
| test_unseen_with_context | 88 | 942 | 0 |

Artefak clean lokal:

- QA bank: `data/pasalid/qa_bank_json_native_expanded_clean.jsonl`
- split: `data/pasalid/json_native_expanded_clean_split/`
- config: `configs/pasalid_experiment_native_expanded_clean_tinyllama.yaml`
- train wrapper: `scripts/train_pasalid_experiment_native_expanded_clean_tinyllama.sh`
- export preset: `tinyllama_native_expanded_clean`

TinyLlama kemudian dilatih ulang pada clean split sampai `1000` iterasi. Training clean tidak lagi memunculkan warning sequence >2048 token. Hasil otomatis `A/B/C/D`:

| Split | A F1 | B F1 | C F1 | D F1 | C - A | D - B | C Citation Component | D Citation Component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seen (`89`) | 0.2618 | 0.4125 | 0.2617 | 0.6518 | -0.0001 | +0.2393 | 0.0309 | 0.3146 |
| unseen (`88`) | 0.2720 | 0.3388 | 0.2210 | 0.4414 | -0.0510 | +0.1025 | 0.0057 | 0.3494 |

Interpretasi clean split:

- Setelah sample panjang dihapus, `D` tetap sangat kuat dan bahkan citation component naik dibanding native-expanded original.
- Adapter-only `C` tidak lagi memberi gain atas `A`; pada unseen tetap lebih buruk daripada baseline no-context.
- Clean split memperkuat kesimpulan bahwa arah paling kuat saat ini adalah **LoRA sebagai context-use adapter (`D`)**, bukan adapter-only memory (`C`).
- Karena clean split menghilangkan warning panjang token, setup ini lebih layak menjadi kandidat eksperimen final TinyLlama daripada native-expanded original.

Review berbantu LLM clean `B` vs `D` pada `20` contoh per split:

| Split | Δ factual D-B | Δ evidence D-B | Δ source D-B | Factual win B/D/tie | Evidence win B/D/tie | Source win B/D/tie |
| --- | ---: | ---: | ---: | --- | --- | --- |
| seen | +0.25 | +0.30 | +0.10 | 2/6/12 | 2/7/11 | 3/4/13 |
| unseen | +0.10 | +0.15 | +0.35 | 4/5/11 | 4/6/10 | 3/7/10 |

Makna review clean:

- Berbeda dari native-expanded original, clean split menunjukkan `D` unggul pada factual/evidence/source baik seen maupun unseen.
- Selisih factual/evidence masih moderat dan banyak tie, sehingga review manual lebih besar tetap dibutuhkan.
- Peningkatan source traceability pada unseen lebih jelas, sejalan dengan citation component otomatis `D` yang naik ke `0.3494`.

### Real-case eval awal

Setelah chat interaktif dicoba, ditemukan bahwa pertanyaan template seperti "dokumen ini mengatur apa" tidak menyerupai pertanyaan user legal yang natural. Untuk menguji robustness awal, dibuat builder `scripts/build_pasalid_realcase_eval.py` yang menghasilkan `60` pertanyaan user-style dari `doc_units`, misalnya pertanyaan tentang dasar aturan, sanksi, definisi, cakupan wilayah, APBN, dan penyelenggaraan pemerintahan daerah.

Artefak lokal:

- QA bank: `data/pasalid/realcase_eval.jsonl`
- split: `data/pasalid/realcase_eval_split/`
- prediction export: `outputs/predictions/pasalid_realcase_eval/`

Hasil TinyLlama clean pada `60` contoh real-case:

| Kondisi | Answer EM | Answer F1 | Citation EM | Citation Component Score |
| --- | ---: | ---: | ---: | ---: |
| A | 0.0000 | 0.1975 | 0.0000 | 0.0000 |
| B | 0.0000 | 0.4852 | 0.0000 | 0.0042 |
| C | 0.0000 | 0.1133 | 0.0000 | 0.0000 |
| D | 0.1500 | 0.6616 | 0.1833 | 0.1875 |

Interpretasi real-case eval:

- `D` tetap menjadi kondisi terbaik pada pertanyaan yang lebih natural, sehingga sinyal context-use adaptation tidak hanya muncul pada template clean.
- `C` jatuh di bawah `A`, sehingga adapter-only memory tidak cukup untuk pertanyaan real-case.
- `B` membaik dengan konteks, tetapi sering tidak disiplin mengeluarkan citation JSON.
- `D` mulai menghasilkan citation yang benar pada sebagian contoh, tetapi masih ada kasus jawaban tidak substantif atau format source_article tidak konsisten.
- Karena real-case eval ini masih heuristic-generated dari doc units, hasilnya perlu diposisikan sebagai audit robustness awal, bukan benchmark final user-facing.

### Natural legal QA LLM-assisted

Untuk mengurangi ketergantungan pada pertanyaan template dan jawaban extractive, dibuat dataset natural legal QA dengan `scripts/build_pasalid_natural_legal_qa.py --use-llm`. Generator menghasilkan pertanyaan user-style dan jawaban parafrase natural, tetapi tetap menyimpan citation terstruktur dan source reference dari `doc_units`. Builder kemudian diberi filter tambahan untuk mengecualikan unit laporan keuangan/report-like yang lebih cocok untuk table/numeric QA daripada legal QA normatif. Split juga dibuat agar held-out law tidak dipilih dari law dengan jumlah row terlalu kecil. Setelah failure audit, builder diperketat lagi untuk menghapus residual APBN/report-like dan menambahkan contoh targeted untuk pasal peralihan/repeal, jumlah kecamatan, provinsi/wilayah hukum, dasar pembentukan, Lembaran Negara, dan pasal pemerintahan daerah yang hanya merujuk peraturan perundang-undangan. Ronde terakhir menambahkan `targeted_slot_repair` yang diprioritaskan untuk provinsi awal pembentukan, nomor/tahun Lembaran Negara, OCR angka kecamatan, dan jawaban repeal yang harus menyebut "dicabut dan dinyatakan tidak berlaku".

Ringkasan dataset:

| Item | Nilai |
| --- | ---: |
| total rows | 541 |
| laws | 17 |
| train rows | 341 |
| valid rows | 39 |
| test seen rows | 121 |
| test unseen rows | 40 |
| answer style | natural paraphrase with structured citation |
| avg max source copy run | 4.2458 |
| max source copy run | 10 |
| report-like rows | 0 |
| targeted slot-repair rows | 79 |
| targeted completeness/transition rows | 135 |

Artefak lokal:

- QA bank: `data/pasalid/qa_bank_natural_legal.jsonl`
- split: `data/pasalid/natural_legal_split/`
- config: `configs/pasalid_natural_legal_tinyllama.yaml`
- adapter: `outputs/adapters/adapters_pasalid_tinyllama_natural_legal.npz`
- prediction export: `outputs/predictions/pasalid_natural_legal/`

Training TinyLlama natural legal pada dataset final targeted-slot-repair selesai sampai `1000` iterasi. Validation loss turun dari `2.074` pada iterasi awal menjadi `1.031` pada iterasi `1000`.

Hasil otomatis `A/B/C/D`:

| Split | A F1 | B F1 | C F1 | D F1 | C - A | D - B | D Citation EM | D Citation Component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seen (`121`) | 0.1807 | 0.2564 | 0.2272 | 0.2886 | +0.0465 | +0.0322 | 0.4959 | 0.5021 |
| unseen (`40`) | 0.1810 | 0.3180 | 0.2381 | 0.3101 | +0.0571 | -0.0079 | 0.5750 | 0.5750 |

Copy dan format metrics:

| Split | Kondisi | Valid JSON | Avg Copy Run | Source 4-gram Copy Ratio | Copy Run >10 |
| --- | --- | ---: | ---: | ---: | ---: |
| seen | B | 0.0000 | 8.7025 | 0.2312 | 0.2975 |
| seen | D | 0.5950 | 6.0496 | 0.1627 | 0.1322 |
| unseen | B | 0.0000 | 13.4000 | 0.3519 | 0.4250 |
| unseen | D | 0.6750 | 6.0000 | 0.2057 | 0.1500 |

Interpretasi natural legal QA:

- Dataset final targeted-slot-repair ini memenuhi target ukuran minimum praktis: train di atas `300` dan test gabungan `161` contoh.
- `D` tetap unggul atas `B` pada seen answer F1 dan jauh lebih baik pada citation/valid JSON, tetapi tidak lagi unggul pada unseen answer F1 setelah slot-repair (`D-B -0.0079`).
- Targeted slot-repair memperbaiki factual slot accuracy unseen dibanding run sebelumnya, tetapi menurunkan margin answer F1 dan tidak menyelesaikan old-province/gazette/repeal secara konsisten.
- `C` tidak memberi manfaat stabil; nilainya hanya mendekati `A` dan jauh di bawah `B/D`, sehingga adapter-only inference tetap tidak layak menjadi kontribusi utama.
- Hasil natural legal QA memperkuat bahwa kontribusi LoRA lebih tepat ditulis sebagai peningkatan disiplin penggunaan konteks/source pada kondisi `D`, bukan sebagai peningkatan answer F1 universal pada semua split.
- Sebelum hasil ini dijadikan klaim final, hasil review LLM perlu dilengkapi audit manual pada contoh gagal untuk membedakan error substansi nyata dari perbedaan parafrase atau format.

Review pairwise berbantu LLM kemudian dilakukan pada `30` contoh seen dan `30` contoh unseen, membandingkan `B` dan `D` secara langsung pada factual correctness, evidence support, source traceability, dan naturalness.

Ringkasan skor rata-rata pairwise:

| Split | Metric | B Avg | D Avg | D - B |
| --- | --- | ---: | ---: | ---: |
| seen | factual correctness | 0.9000 | 0.8333 | -0.0667 |
| seen | evidence support | 0.9667 | 0.9333 | -0.0333 |
| seen | source traceability | 0.1000 | 1.0000 | +0.9000 |
| seen | naturalness | 0.6667 | 0.8667 | +0.2000 |
| unseen | factual correctness | 1.0667 | 0.7333 | -0.3333 |
| unseen | evidence support | 1.2000 | 0.8333 | -0.3667 |
| unseen | source traceability | 0.0000 | 1.1000 | +1.1000 |
| unseen | naturalness | 0.8000 | 0.8667 | +0.0667 |

Winner count pairwise:

| Split | Metric | B Wins | D Wins | Ties |
| --- | --- | ---: | ---: | ---: |
| seen | factual correctness | 9 | 8 | 13 |
| seen | evidence support | 10 | 8 | 12 |
| seen | source traceability | 3 | 17 | 10 |
| seen | naturalness | 9 | 13 | 8 |
| seen | overall | 13 | 16 | 1 |
| unseen | factual correctness | 12 | 6 | 12 |
| unseen | evidence support | 12 | 6 | 12 |
| unseen | source traceability | 0 | 17 | 13 |
| unseen | naturalness | 11 | 11 | 8 |
| unseen | overall | 14 | 15 | 1 |

Makna review pairwise natural legal QA:

- Pada seen, `D` masih unggul tipis overall karena traceability dan naturalness, tetapi factual/evidence tidak lagi unggul setelah slot-repair.
- Pada unseen, `D` tetap unggul tipis overall dan source traceability, tetapi factual/evidence jelas memihak `B` pada sample review.
- `B` hampir selalu dinilai `source-missing` dan sering `too-extractive`, sehingga F1 tinggi tidak otomatis berarti jawaban siap dipakai sebagai legal QA akuntabel.
- `D` lebih traceable dan lebih natural, tetapi masih sering mendapat label `factually-wrong` atau `unsupported-answer`; ini menunjukkan adapter natural legal perlu perbaikan substansi, bukan hanya format.
- Temuan final untuk natural QA sebaiknya ditulis sebagai tradeoff yang lebih kuat: LoRA + context memperbaiki traceability, naturalness, valid JSON, dan copy-rate, tetapi answer F1 dan factual correctness held-out law tidak stabil setelah targeted slot-repair.

Audit targeted kemudian dilakukan pada failure cases `D` dari pairwise review. Setelah targeted slot-repair, audit ini dibuat reproducible dengan `scripts/summarize_pasalid_failure_audit.py`, yang membaca label review dan reason untuk mengelompokkan sisa failure.

Ringkasan jumlah failure `D`:

| Split | Reviewed Rows | Broad Failure Rows | Strict Failure Rows | Strict Failure Rate |
| --- | ---: | ---: | ---: | ---: |
| seen | 30 | 27 | 21 | 0.7000 |
| unseen | 30 | 23 | 20 | 0.6667 |

`Broad failure` mencakup skor factual/evidence kurang dari `2` atau label error kuat. `Strict failure` hanya menghitung label `factually-wrong`, `unsupported-answer`, `source-wrong`, atau `source-missing`.

Kategori error utama `D` dari audit targeted:

| Kategori | Seen | Unseen | Pola |
| --- | ---: | ---: | --- |
| substantive factual/evidence error | 11 | 16 | Jawaban masih salah atau unsupported pada sebagian query walau traceability naik |
| entity/count/list confusion | 13 | 14 | Pertanyaan provinsi asal, jumlah kecamatan, Lembaran Negara, dan daftar wilayah masih rentan |
| incomplete or wrong focus | 18 | 17 | Model menjawab tanggal saat ditanya provinsi, atau tidak menyebut detail kunci seperti nomor Lembaran Negara |
| source/format error | 17 | 14 | JSON/source raw masih kadang hilang/salah meskipun constrained post-process bisa memperbaikinya |
| too extractive | 15 | 15 | Jawaban masih sering menyalin frasa dokumen atau daftar mentah |
| prompt/instruction echo or unnatural | 16 | 11 | Masih ada echo instruksi/referensi dan kalimat janggal |
| polarity/transition confusion | 6 | 0 | Error peralihan/repeal masih muncul pada seen; unseen lebih didominasi entity/slot failures |

Makna audit failure cases:

- Targeted slot-repair membuat `D` tetap unggul tipis overall karena traceability/naturalness, tetapi strict failure masih tinggi sehingga hasil tidak boleh dibaca sebagai solved legal QA.
- Banyak kegagalan `D` tetap error substansi nyata, bukan sekadar perbedaan parafrase dengan gold answer.
- Source/citation dan factual correctness harus dilaporkan terpisah: source dapat diperbaiki dengan constraint, sedangkan isi jawaban belum otomatis benar.
- Pertanyaan held-out law yang melibatkan provinsi asal pembentukan, angka kecamatan, Lembaran Negara, dan daftar wilayah masih menjadi residual bottleneck.
- Error transition/repeal lebih terkendali dibanding audit sebelumnya, terutama pada unseen split.
- Next action teknis paling relevan adalah validasi isi jawaban berbasis extraction/checker untuk angka, provinsi, dan status repeal, bukan hanya menambah post-processing JSON/source.

Untuk memisahkan answer F1 dari correctness fakta spesifik, ditambahkan `scripts/eval_pasalid_natural_slots.py`. Evaluator ini membaca `prompt`, `gold`, dan `prediction`, lalu membuat slot check untuk kasus yang dapat diekstraksi secara deterministik: jumlah kecamatan, daftar kecamatan, provinsi saat pembentukan awal, provinsi saat ini, Lembaran Negara, status repeal, dan keberlakuan aturan peralihan.

Hasil factual slot evaluation raw `B` vs `D`:

| Split | Kondisi | Slot Checks | Correct | Slot Accuracy |
| --- | --- | ---: | ---: | ---: |
| seen | B raw | 44 | 25 | 0.5682 |
| seen | D raw | 44 | 22 | 0.5000 |
| unseen | B raw | 21 | 10 | 0.4762 |
| unseen | D raw | 21 | 9 | 0.4286 |

Breakdown slot `D` raw:

| Split | Slot | Correct / Total | Accuracy |
| --- | --- | ---: | ---: |
| seen | current province | 2/2 | 1.0000 |
| seen | gazette reference | 2/6 | 0.3333 |
| seen | kecamatan count | 11/12 | 0.9167 |
| seen | kecamatan list | 0/2 | 0.0000 |
| seen | old formation province | 0/3 | 0.0000 |
| seen | repeal status | 3/11 | 0.2727 |
| seen | transition validity | 4/8 | 0.5000 |
| unseen | current province | 1/1 | 1.0000 |
| unseen | gazette reference | 1/4 | 0.2500 |
| unseen | kecamatan count | 3/4 | 0.7500 |
| unseen | kecamatan list | 0/1 | 0.0000 |
| unseen | old formation province | 1/4 | 0.2500 |
| unseen | repeal status | 1/4 | 0.2500 |
| unseen | transition validity | 2/3 | 0.6667 |

Makna factual slot evaluation:

- Pada seen split, `D` lebih rendah daripada `B` pada slot accuracy (`0.5000` vs `0.5682`), terutama karena list extraction, old formation province, dan repeal status masih rapuh.
- Pada unseen split, slot-repair menaikkan `D` dari `0.1765` ke `0.4286`, tetapi `B` masih sedikit lebih tinggi (`0.4762`); kasus Kabupaten Tanah Datar tetap gagal pada provinsi awal, Lembaran Negara, dan daftar kecamatan.
- Constrained output tidak mengubah slot accuracy (`B` dan `D` sama dengan raw untuk slot ini), sehingga post-processing source/JSON memang tidak memperbaiki substansi fakta.
- Slot evaluation menguatkan batas klaim: `D` baik untuk traceability dan beberapa factual slots seen, tetapi belum reliable untuk factual slot generalization pada held-out law.

Sebagai tindak lanjut decoding/format constraint, ditambahkan `scripts/postprocess_pasalid_natural_predictions.py`. Script ini tidak mengubah model, tetapi membuat varian constrained dengan cara:

- mengambil `answer` dari JSON jika ada, atau membersihkan teks mentah jika JSON tidak valid;
- menghapus echo `Referensi:`, `Instruksi:`, atau `Q:` dari jawaban;
- memaksa output kembali ke schema JSON `answer + source_*` memakai referensi dari prompt sumber.

Evaluator natural juga ditambah metrik pendukung `answer_precision`, `answer_recall`, `short_answer_rate`, `prompt_echo_rate`, dan `instruction_echo_rate` agar failure completeness tidak tertutup oleh F1 saja.

Hasil constrained otomatis:

| Split | Kondisi | Answer F1 | Answer Recall | Citation EM | Valid JSON | Prompt Echo | Instruction Echo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seen | B raw | 0.2564 | 0.2781 | 0.0000 | 0.0000 | 0.1653 | 0.1901 |
| seen | B constrained | 0.2592 | 0.2503 | 1.0000 | 1.0000 | 0.0083 | 0.0248 |
| seen | D raw | 0.2886 | 0.2780 | 0.4959 | 0.5950 | 0.1240 | 0.2810 |
| seen | D constrained | 0.2848 | 0.2557 | 1.0000 | 1.0000 | 0.0165 | 0.1570 |
| unseen | B raw | 0.3180 | 0.3361 | 0.0000 | 0.0000 | 0.2000 | 0.2500 |
| unseen | B constrained | 0.3210 | 0.3023 | 1.0000 | 1.0000 | 0.0000 | 0.0250 |
| unseen | D raw | 0.3101 | 0.2836 | 0.5750 | 0.6750 | 0.0250 | 0.0750 |
| unseen | D constrained | 0.3051 | 0.2789 | 1.0000 | 1.0000 | 0.0000 | 0.0500 |

Review pairwise constrained `30` contoh per split belum direrun setelah slot-repair. Angka berikut adalah run constrained sebelumnya dan hanya dipakai sebagai bukti bahwa forcing JSON/source dapat menghilangkan perbedaan traceability secara desain:

| Split | Metric | B Wins | D Wins | Ties |
| --- | --- | ---: | ---: | ---: |
| seen | factual correctness | 9 | 9 | 12 |
| seen | evidence support | 9 | 9 | 12 |
| seen | source traceability | 2 | 0 | 28 |
| seen | naturalness | 11 | 10 | 9 |
| seen | overall | 12 | 11 | 7 |
| unseen | factual correctness | 9 | 9 | 12 |
| unseen | evidence support | 7 | 9 | 14 |
| unseen | source traceability | 2 | 0 | 28 |
| unseen | naturalness | 8 | 11 | 11 |
| unseen | overall | 11 | 13 | 6 |

Makna constrained run:

- Constraint efektif untuk format: valid JSON dan citation EM menjadi `1.0` pada metrik otomatis karena source dipaksa dari prompt.
- Constraint mengurangi prompt/instruction echo, tetapi recall jawaban tetap turun sedikit; ini menunjukkan failure completeness adalah masalah substansi generatif, bukan hanya parsing output.
- Setelah `B` dan `D` sama-sama diberi format/source constraint, keunggulan source traceability `D` hilang secara desain; factual/evidence menjadi seimbang, dengan `D` masih unggul tipis pada unseen overall.
- Karena itu, klaim final perlu memisahkan tiga hal: raw model behavior (`D` lebih disiplin sumber daripada `B`), constrained system behavior (format bisa dipaksa untuk semua kondisi context), dan answer completeness (masih membutuhkan data/decoding tambahan).

Benchmark efisiensi clean `A/B/C/D` kemudian diperbaiki menjadi per-example dan tokenizer-based. Script `scripts/benchmark_pasalid_experiment.py` sekarang mendukung `--per-example`, menghitung prompt token dengan tokenizer model, dan mengukur latency tiap contoh sehingga p50/p95 tidak lagi hasil pembagian rata dari satu batch export.

Benchmark natural legal QA final juga dijalankan dengan preset `tinyllama_natural_legal`, split `data/pasalid/natural_legal_split/`, `20` contoh per split, mode `--per-example`, dan `--max-new-tokens 96`.

Hasil natural legal QA:

| Split | Kondisi | Avg prompt tokens | Latency avg | Latency p50 | Latency p95 | Generated tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| seen | A | 40.2 | 4.6856 | 4.6739 | 4.8243 | 20.4989 |
| seen | B | 224.6 | 4.7001 | 5.0188 | 5.2113 | 15.2868 |
| seen | C | 40.2 | 9.7087 | 9.6954 | 10.2435 | 7.9568 |
| seen | D | 224.6 | 7.1523 | 7.4189 | 10.0381 | 9.9898 |
| unseen | A | 33.0 | 3.6250 | 4.6403 | 4.7000 | 18.9930 |
| unseen | B | 260.4 | 4.2514 | 4.9996 | 5.2415 | 16.4651 |
| unseen | C | 33.0 | 9.8435 | 9.7575 | 10.5666 | 5.9633 |
| unseen | D | 260.4 | 7.6658 | 8.3172 | 10.2440 | 9.2619 |

Makna benchmark natural legal QA:

- `C` memang memakai prompt paling pendek, tetapi latency paling buruk karena adapter inference membuat decoding lebih lambat.
- `D` memakai prompt sepanjang `B` dan menambah biaya latency sekitar `+2.45s` pada seen serta `+3.41s` pada unseen.
- `B` menjadi baseline efisiensi terbaik untuk retrieval-only context: prompt lebih panjang daripada `A/C`, tetapi latency tetap dekat dengan `A` dan jauh lebih cepat daripada `D`.
- Nilai kualitas/traceability `D` harus dibaca sebagai tradeoff: source discipline dan naturalness naik, tetapi runtime lebih mahal dibanding `B`.
- Peak RSS masih proxy proses in-process dan belum cukup isolatif untuk klaim memory final.

Hasil pada `10` contoh per split:

| Split | Kondisi | Avg prompt tokens | Latency avg | Latency p50 | Latency p95 | Generated tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| seen | A | 53.6 | 5.8790 | 5.7993 | 6.1721 | 21.9764 |
| seen | B | 193.4 | 6.5312 | 6.4886 | 6.8189 | 18.2049 |
| seen | C | 53.6 | 10.2307 | 11.7526 | 11.9534 | 10.8107 |
| seen | D | 193.4 | 9.4996 | 10.1627 | 12.2890 | 9.4215 |
| unseen | A | 60.1 | 5.8786 | 5.8039 | 6.1506 | 22.1140 |
| unseen | B | 291.5 | 6.5866 | 6.6209 | 7.0114 | 19.5548 |
| unseen | C | 60.1 | 11.0282 | 11.8227 | 11.9290 | 9.5845 |
| unseen | D | 291.5 | 9.8902 | 12.2894 | 12.7186 | 9.7066 |

Makna benchmark per-example:

- `C` memang mengurangi prompt token drastis dibanding `B/D`, tetapi latency tetap lebih buruk daripada `A/B` karena adapter inference lebih mahal.
- `D` membawa kualitas paling tinggi, tetapi memakai prompt sepanjang `B` dan latency lebih tinggi daripada `B`.
- `B` masih menjadi baseline efisiensi-kualitas yang kuat: prompt lebih panjang dari `A/C`, tetapi latency tetap rendah karena tidak memuat adapter.
- Peak RSS masih proxy proses dan belum isolasi sempurna per kondisi; klaim memory final tetap perlu prosedur terpisah jika ingin dilaporkan kuat.

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

- TinyLlama dan Mistral q4 long memberi sinyal bahwa LoRA dapat bertindak sebagai context-use/domain adapter.
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

### Ranking praktis pada pilot JSON-large

Dalam setup pilot JSON `answer + source` A/B/C lama, ranking praktisnya adalah:

1. `TinyLlama`
2. `Qwen3`
3. `Mistral q4`

Ranking ini spesifik untuk pilot lama:

- corpus Pasal.id-derived saat ini
- `json_large_split/`
- format jawaban yang sedang diuji
- checkpoint adapter yang saat ini tersedia

Catatan: ranking ini tidak boleh dibaca sebagai ranking final seluruh eksperimen karena clean native-expanded dan kondisi `D` mengubah fokus utama menjadi context-use adaptation pada TinyLlama.

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
- `D` memperlihatkan sinyal context-use adaptation pada TinyLlama dan Mistral q4 long, tetapi tidak pada Qwen3; pada clean TinyLlama, `D` menjadi cabang QA paling kuat.
- Kualitas jawaban dan kualitas keterlacakan sumber saat ini terpisah secara empiris: jawaban bisa membaik tanpa diikuti attribution yang baik.
- Memperbesar model tidak otomatis memperbaiki masalah traceability.

## Temuan yang Stabil

- Kondisi `B` tetap menjadi upper bound praktis untuk adapter-only inference (`C`) di seluruh eksperimen yang sudah dijalankan.
- Kondisi `C` dalam beberapa setup memberi peningkatan atas `A`, tetapi hasil clean menunjukkan internalisasi adapter-only belum stabil.
- Kondisi `D` layak menjadi cabang QA utama terbaru karena clean TinyLlama menunjukkan gain besar atas `B` pada seen dan unseen.
- Format JSON `answer + source` adalah format eksperimen paling layak saat ini untuk menjaga kualitas jawaban sambil tetap memungkinkan evaluasi traceability.
- `TinyLlama` tetap menjadi baseline paling kuat untuk konfigurasi clean yang sudah diuji mendalam.

## Metrik yang Masih Belum Bergerak

- `EM` hampir selalu tetap `0`.
- `Citation EM` hampir selalu tetap `0`.
- `Citation Component Score` masih lemah pada A/B/C, tetapi mulai bergerak pada `D` clean TinyLlama.

Makna dari pola ini adalah bahwa peningkatan performa saat ini lebih banyak terjadi pada kualitas isi jawaban dan penggunaan konteks daripada pada kedisiplinan model dalam menyebut sumber secara andal.

## Batas Klaim yang Aman

- Aman untuk menyatakan bahwa adapter-based internalization dapat meningkatkan kualitas jawaban dibanding kondisi no-context pada beberapa setup, tetapi tidak stabil pada clean split.
- Aman untuk menyatakan bahwa pendekatan context-based masih lebih kuat daripada adapter-only inference dalam eksperimen ini.
- Aman untuk menyatakan bahwa adapter dengan konteks (`D`) memberi sinyal positif pada sebagian model dan menjadi hasil QA terkuat pada clean TinyLlama, tetapi efeknya belum universal.
- Belum aman untuk menyatakan bahwa model sudah mampu memberi source attribution yang reliabel dan machine-checkable tanpa intervensi tambahan.

## Bottleneck Utama Saat Ini

- source traceability masih menjadi kelemahan utama di seluruh model dan format yang diuji
- sebagian sample masih cukup panjang dan berat untuk model kecil
- kualitas QA bank sudah cukup untuk baseline, tetapi masih dapat diperbaiki lebih lanjut jika eksperimen akan diperluas

## Arah Lanjutan yang Paling Rasional

1. pertahankan clean native-expanded JSON `answer + source` sebagai kandidat setup QA final
2. posisikan `D` sebagai cabang utama context-use adaptation dan `C` sebagai stress test adapter-only internalization
3. perlakukan source-components sebagai hasil ablasi, bukan jalur utama lanjutan
4. jika eksperimen diteruskan, arah paling bernilai adalah review final B/D, task khusus source attribution atau citation prediction, dan validasi lintas model, bukan sekadar memperbesar model lagi

## Rekomendasi Setup Utama Tesis

Berdasarkan seluruh eksperimen yang sudah dijalankan, setup yang paling layak diposisikan sebagai **eksperimen utama tesis** saat ini adalah sebagai berikut.

### Eksperimen utama 1: QA context-use adaptation

- objective: menguji apakah adapter LoRA dapat membantu model menggunakan dokumen sumber secara lebih efektif daripada baseline context-only
- dataset format: clean native-expanded JSON `answer + source`
- model utama: `TinyLlama`
- kondisi utama yang dilaporkan:
  - `A`: base tanpa konteks
  - `B`: base dengan konteks dokumen
  - `C`: base + adapter tanpa konteks
  - `D`: base + adapter + konteks dokumen
- metrik utama:
  - `F1`
- metrik tambahan:
  - `EM`
  - `Citation EM`
  - `Citation Component Score`
  - metrik efisiensi inferensi

Alasan pemilihan:

- Clean native-expanded TinyLlama memberi hasil QA paling kuat pada `D`, tanpa warning sequence >2048 token.
- Kondisi `C` tetap penting sebagai stress test adapter-only internalization, tetapi bukan klaim utama karena tidak stabil pada clean split.
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
- review manual/LLM seed diposisikan sebagai bukti awal operasional untuk akuntabilitas jawaban, bukan hasil final.

## Pemisahan Bukti Utama dan Bukti Pendukung

### Bukti utama

- hasil A/B/C/D TinyLlama pada clean native-expanded JSON `answer + source`
- hasil source prediction Mistral q4
- benchmark efisiensi inferensi per-example untuk A/B/C/D

### Bukti pendukung

- eksperimen Qwen3
- eksperimen Mistral pada QA utama
- hasil JSON-large pilot
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

Benchmark awal TinyLlama pada setup JSON-large `answer + source`, seen split, `10` contoh menghasilkan ringkasan berikut:

| Kondisi | Avg Prompt Token Proxy | Avg Latency (s) | p50 (s) | p95 (s) | Peak RSS Proxy (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 27.4 | 2.3434 | 2.3434 | 2.3434 | 1201733632 |
| B | 110.2 | 2.3609 | 2.3609 | 2.3609 | 2662760448 |
| C | 27.4 | 3.9444 | 3.9444 | 3.9444 | 2652045312 |

Interpretasi:

- Ini adalah benchmark proxy historis; benchmark clean per-example pada bagian native-expanded clean lebih layak dipakai untuk pembahasan final.
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

- target utama terbaru: kualitas jawaban dengan adapter yang membantu penggunaan konteks dokumen (`D`)
- stress test tambahan: kualitas jawaban tanpa dokumen sumber saat inferensi (`C`)
- hasil clean TinyLlama: `D > B > A > C` pada unseen dan `D > B > A ~= C` pada seen
- model baseline terbaik yang sudah diuji mendalam: `TinyLlama`
- bottleneck utama: source traceability

### Ringkasan cabang source prediction

- target utama: prediksi sumber hukum secara eksplisit
- semua model menghasilkan metrik yang tidak lagi nol
- `Mistral q4` terbaik pada `source_component_score`
- `Qwen3` terbaik pada `source_exact_match`

### Kesimpulan lintas cabang

- kemampuan menghasilkan jawaban dan kemampuan memberi attribution sumber adalah dua kemampuan yang berbeda.
- eksperimen saat ini mendukung klaim terbatas bahwa internalisasi isi jawaban dapat terjadi tanpa konteks penuh pada beberapa setup, tetapi clean split menunjukkan cabang ini tidak stabil.
- eksperimen saat ini lebih kuat mendukung klaim bahwa LoRA dapat berperan sebagai context-use adapter ketika dokumen sumber tetap diberikan.
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
