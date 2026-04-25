# ✅ Checklist Kesesuaian Eksperimen dengan Target Proposal

Dokumen ini memetakan target proposal terhadap status eksperimen yang sudah dibangun di repo `lora-mlx`, sekaligus menjadi rencana kerja untuk mengubah hasil pilot menjadi eksperimen final tesis.

## Keterangan Status

| Emoji | Status | Arti |
| --- | --- | --- |
| ✅ | done | sudah cukup kuat untuk dipakai sebagai jawaban penelitian sementara |
| 🟡 | partial | sudah ada bukti awal, tetapi belum cukup kuat untuk klaim final tesis |
| ❌ | not yet | belum cukup terisi atau belum diukur secara sistematis |

## 1) Target Inti Proposal

| Target | Status | Bukti Saat Ini | Gap Utama | Next Action |
| --- | --- | --- | --- | --- |
| Apakah adapter LoRA dapat menginternalisasi informasi inti dokumen panjang sehingga model tetap dapat menjawab tanpa dokumen sumber saat inferensi? | 🟡 | Di beberapa setup, kondisi `C` mengungguli `A`; pola `B > C > A` muncul berulang pada QA utama | Belum ada eksperimen final tunggal dengan setup yang benar-benar dikunci | Kunci satu setup final dan jalankan evaluasi final pada dataset utama yang lebih besar |
| Model mana yang paling menjanjikan untuk QA utama? | 🟡 | TinyLlama paling stabil dan paling baik pada QA utama; Qwen3 di bawah TinyLlama; Mistral di bawah keduanya | Ranking masih berbasis pilot setup | Validasi ranking pada setup final yang lebih besar |
| Model mana yang paling menjanjikan untuk source attribution? | 🟡 | TinyLlama source component `0.5357`; Mistral `0.7262`; Qwen3 exact match `0.3810` | Masih berbasis baseline awal source branch | Jalankan source branch final dan bandingkan di setup yang dikunci |

## 2) Kesesuaian terhadap Metrik Proposal

| Metrik / Aspek | Status | Kondisi Sekarang | Catatan | Next Action |
| --- | --- | --- | --- | --- |
| F1 sebagai nilai utama pembuktian | ✅ | F1 paling informatif di seluruh eksperimen QA utama | Sudah layak jadi metrik utama tesis | Pertahankan sebagai metrik utama |
| EM sebagai pendukung kualitas jawaban | ✅ | EM sudah dihitung konsisten, tetapi hampir selalu `0` | Tetap berguna sebagai metrik ketat tambahan | Tetap dilaporkan, tapi bukan metrik utama |
| Evidence attribution | 🟡 | Pada QA utama masih lemah; pada source prediction sudah mulai bergerak jelas | Source-prediction masih perlu test set yang tidak menyebut sumber eksplisit | Buat varian source-prediction `explicit` dan `implicit` |
| Evidence support rate | 🟡 | Rubrik dan seed manual review sudah berjalan; review semi-otomatis `A/B/C` dan pairwise `B/D` sudah ada | Belum ada review final yang balanced untuk `A/B/C/D` | Review manual minimal `30-50` contoh per kondisi |
| Unsupported answer rate | 🟡 | Kategori `unsupported-answer` dan `factually-wrong` sudah dihitung pada seed review | Belum cukup besar untuk klaim final | Coding manual balanced pada subset evaluasi final |
| Ketepatan rujukan pasal atau bagian dokumen | 🟡 | QA utama masih lemah pada citation metrics; source prediction branch sudah bermakna | Citation di QA utama belum stabil | Pertahankan source branch sebagai pembanding attribution |
| Jumlah token konteks saat inferensi | ✅ | Benchmark clean dan natural QA sudah memakai token tokenizer model; `C` mengurangi prompt token dibanding `B/D` | Token dihitung pada subset evaluasi benchmark, bukan seluruh test set | Laporkan ukuran subset bersama angka token |
| Latensi p50 / p95 | ✅ | Benchmark clean dan natural QA sudah mendukung mode per-example; natural QA dijalankan pada `20` contoh per split | Jika akan jadi angka final tesis, angka dapat diperbesar lagi | Laporkan sebagai benchmark subset terkontrol |
| Penggunaan memori | 🟡 | Peak RSS proxy sudah dicatat pada benchmark awal | Definisi dan prosedur ukur belum final | Tetapkan prosedur memory measurement yang konsisten |

## 3) Kesesuaian terhadap Tujuan Khusus Proposal

| Tujuan Proposal | Status | Bukti Saat Ini | Gap Utama | Next Action |
| --- | --- | --- | --- | --- |
| Merancang artefak internalisasi dokumen ke adapter LoRA | ✅ | Pipeline artefak sudah ada: ingestion, doc units, QA generation, split A/B/C, training, evaluation, reporting | - | Pertahankan dan dokumentasikan |
| Menetapkan rancangan dataset eksperimen yang terukur dan dapat direplikasi | ✅ | Natural legal QA final targeted sudah `535` row, train `338`, test gabungan `158`, dan pipeline generate/train/eval/review terdokumentasi | Coverage regulasi masih terbatas pada corpus Pasal.id lokal | Laporkan batas coverage dan jangan overclaim sebagai benchmark nasional |
| Membandingkan performa A/B/C/D pada metrik utama dan pendukung | ✅ | A/B/C/D sudah dibandingkan pada clean split dan natural legal QA final; pairwise review `B/D` sudah `30` seen + `30` unseen; efisiensi natural QA sudah diukur per-example | Review manual belum mencakup A/C | Tambah audit A/C hanya jika dibutuhkan untuk lampiran |
| Menganalisis sejauh mana C mendekati B dan melampaui A | 🟡 | Pola `B > C > A` muncul di beberapa setup yang stabil | Belum ada satu eksperimen final sebagai basis klaim utama | Kunci satu protokol final |
| Merumuskan rekomendasi desain sistem yang seimbang antara kualitas, traceability, dan efisiensi | ✅ | Rekomendasi sudah didukung kualitas, traceability, copy-rate, dan efisiensi: `D` lebih disiplin sumber tetapi lebih lambat dari `B` | Memory masih proxy proses | Jika diperlukan, tambah prosedur memory terisolasi |

## 4) Ringkasan Kesiapan Saat Ini

### ✅ Sudah Cukup Kuat

| Aspek | Status Ringkas |
| --- | --- |
| Artefak penelitian dan pipeline eksperimen | Sudah ada dan berjalan |
| F1 sebagai metrik utama | Sudah kuat |
| EM sebagai metrik tambahan | Sudah jelas posisinya |
| Sinyal internalisasi parsial pada task QA utama | Sudah ada |
| Hasil source attribution bermakna pada branch source prediction | Sudah ada |

### 🟡 Sudah Kuat sebagai Hasil Pilot

| Aspek | Status Ringkas |
| --- | --- |
| Ranking model sementara untuk QA utama | Sudah terbaca, belum final |
| Ranking model sementara untuk source prediction | Sudah terbaca, belum final |
| Bottleneck utama pada source traceability | Sudah teridentifikasi jelas |

### ❌ Belum Cukup Kuat untuk Klaim Final Tesis

| Aspek | Kenapa Belum |
| --- | --- |
| Evidence support rate | Sudah ada seed, tetapi belum balanced dan belum cukup besar |
| Unsupported answer rate | Sudah ada seed, tetapi belum balanced dan belum cukup besar |
| Latency p50/p95 | Sudah ada benchmark awal, tetapi p50/p95 masih proxy |
| Penggunaan memori | Sudah ada peak RSS proxy, tetapi prosedur belum dikunci |
| Validasi final pada setup utama yang dikunci | Belum ada satu eksperimen final utama |

## 5) Risiko Metodologis yang Harus Diperbaiki

| Risiko | Dampak terhadap Klaim | Perbaikan |
| --- | --- | --- |
| Pertanyaan source-prediction sering menyebut UU, tahun, atau pasal secara eksplisit | Metrik source prediction dapat terbaca sebagai ekstraksi dari prompt, bukan internalisasi attribution | Pisahkan source-prediction menjadi `explicit` dan `implicit`; jadikan `implicit` sebagai bukti utama |
| Pertanyaan QA utama sebagian menyebut sumber hukum eksplisit | Citation score bisa kurang bermakna karena sumber sudah tersurat | Untuk evaluasi traceability, buat subset pertanyaan tanpa sumber eksplisit |
| Seen split lebih kuat daripada unseen split | Klaim internalisasi bisa terbaca sebagai memorization pada dokumen training | Laporkan seen dan unseen secara terpisah; jangan menggabungkan sebagai satu skor utama |
| Ukuran test set masih kecil | Ranking model dan selisih `A/B/C` belum stabil secara tesis | Perbesar test seen dan unseen sebelum eksperimen final |
| Manual review belum balanced antar kondisi | Evidence support rate dan unsupported answer rate belum dapat dibandingkan adil | Review jumlah contoh yang sama untuk `A`, `B`, `C`, dan `D` |
| Benchmark efisiensi masih proxy | Klaim efisiensi belum kuat | Ukur latency per contoh, token tokenizer, generated tokens/sec, dan peak memory dengan prosedur tetap |

## 6) Rencana Eksperimen Final

### Cabang 1: QA Internalization

| Komponen | Keputusan Final |
| --- | --- |
| Objective | Menguji apakah adapter LoRA dapat meningkatkan kualitas jawaban tanpa dokumen sumber saat inferensi |
| Model utama | `TinyLlama` |
| Model pembanding | `Qwen3`, `Mistral q4` sebagai pendukung, bukan basis klaim utama |
| Format data | JSON `answer + source` |
| Kondisi | `A`: base no-context; `B`: base with-context; `C`: adapter no-context; `D`: adapter with-context sebagai cabang tambahan |
| Split wajib | `test_seen` dan `test_unseen` dilaporkan terpisah |
| Metrik utama | `Answer F1` |
| Metrik pendukung | `Answer EM`, `Citation EM`, `Citation Component Score`, manual review, efisiensi inferensi |
| Klaim aman yang ditargetkan | `C` dapat mengungguli `A` pada kualitas jawaban di sebagian setup, `B` tetap upper bound untuk adapter-only inference, dan `D` dapat membantu penggunaan konteks pada sebagian model |

Dataset target minimal untuk eksperimen final:

| Split | Target Minimal | Catatan |
| --- | ---: | --- |
| Train | `300-500` contoh | Tetap document-grounded untuk training adapter |
| Valid | `50` contoh | Dipakai untuk monitoring, bukan klaim utama |
| Test seen | `100` contoh | Pertanyaan dari regulasi yang ada di train, tetapi tidak literal duplicate |
| Test unseen | `100` contoh | Regulasi held-out penuh |

### Cabang 2: Source Attribution

| Komponen | Keputusan Final |
| --- | --- |
| Objective | Menguji apakah rujukan sumber hukum dapat diprediksi ketika attribution dijadikan task eksplisit |
| Model utama | `Mistral q4` untuk `source_component_score` |
| Model pembanding | `Qwen3` untuk `source_exact_match`, `TinyLlama` sebagai baseline ringan |
| Format data | JSON `source_type`, `source_number`, `source_year`, `source_article` |
| Varian evaluasi | `source-explicit` dan `source-implicit` |
| Bukti utama | `source-implicit`, karena prompt tidak boleh menyebut UU/tahun/pasal target secara langsung |
| Metrik utama | `source_component_score` dan `source_exact_match` |
| Metrik pendukung | valid JSON rate dan akurasi per komponen |

Aturan dataset source attribution:

| Aturan | Tujuan |
| --- | --- |
| `source-explicit` boleh menyebut UU/pasal | Sanity check parsing dan format |
| `source-implicit` tidak boleh menyebut UU, nomor, tahun, atau pasal target | Menguji attribution yang lebih dekat ke internalisasi |
| Test set harus split by law atau by article group | Mengurangi leakage antar train dan test |
| Laporkan `explicit` dan `implicit` terpisah | Menghindari overclaim dari test yang terlalu mudah |

Status implementasi awal:

| Item | Status | Catatan |
| --- | --- | --- |
| Builder menghasilkan subset `explicit` dan `implicit` | ✅ | `scripts/build_pasalid_source_dataset.py` sekarang menulis `data/pasalid_source/explicit/` dan `data/pasalid_source/implicit/` |
| Detector pertanyaan eksplisit | ✅ | Pertanyaan ditandai eksplisit jika menyebut UU/nomor/tahun/pasal target |
| Augmentasi pertanyaan `implicit` | ✅ | Builder membuat pertanyaan implisit dari isi jawaban sehingga tidak bergantung pada pertanyaan asli yang sering menyebut sumber |
| Ukuran subset `implicit` | 🟡 | Naik menjadi `238` row, dengan `24` row pada test law saat split sekarang |
| Validasi leakage `implicit` | ✅ | Detector menemukan `0` target-source mention pada train/valid/test `implicit` |
| Evaluasi awal `implicit` | ✅ | Mistral q4 menjadi yang terbaik pada `source_component_score` (`0.5312`), tetapi semua model `source_exact_match = 0` |
| Split `seen/unseen` untuk source attribution | ✅ | `implicit` sekarang memiliki `test_seen` (`71` row) dan `test_unseen` (`24` row) |
| Retraining Mistral q4 pada `implicit` | 🟡 | Belum memperbaiki component score total; adapter lama masih lebih tinggi pada seen dan unseen |
| Kesiapan evaluasi final `implicit` | 🟡 | Sudah lebih layak untuk evaluasi awal; untuk tesis final tetap perlu test implicit lebih besar, lebih beragam, dan objective attribution yang lebih kuat |

### Manual Review Final

| Komponen | Target |
| --- | --- |
| Jumlah contoh | Minimal `30-50` contoh per kondisi `A/B/C/D` |
| Sampling | Balanced dari test seen dan test unseen |
| Dimensi | `factual_correctness`, `evidence_support`, `source_traceability` |
| Rate final | `evidence_support_rate`, `unsupported_answer_rate`, `source_traceability_rate` |
| Posisi LLM judge | Alat bantu coding awal; subset tetap diverifikasi manual |

### Benchmark Efisiensi Final

| Metrik | Perbaikan yang Dibutuhkan |
| --- | --- |
| Prompt token | Gunakan tokenizer model, bukan `text.split()` |
| Latency avg/p50/p95 | Ukur per contoh dengan warm-up terpisah |
| Generated tokens/sec | Catat jumlah token output dan waktu decoding |
| Memory | Gunakan prosedur peak memory yang sama untuk `A`, `B`, dan `C` |
| Interpretasi | Pisahkan pengurangan panjang konteks dari efisiensi runtime murni |

## 7) Prioritas Eksekusi

| Prioritas | Pekerjaan | Alasan |
| --- | --- | --- |
| 1 | Perbaiki residual OCR/report-like filtering dan error negasi natural QA | Failure audit menunjukkan masih ada semantic drift, question echo, wrong polarity, dan sisa unit report-like |
| 2 | Perbesar benchmark efisiensi pada setup natural QA | Mengisi dimensi efisiensi dengan data yang sesuai task final |
| 3 | Perbesar source attribution `implicit` jika attribution tetap jadi sub-eksperimen | Test implicit saat ini sudah lebih baik, tetapi masih perlu coverage lebih besar untuk klaim final |
| 4 | Uji objective attribution yang lebih kuat | Retraining implicit biasa belum memperbaiki component score total |
| 5 | Susun pembahasan tesis dengan dua cabang eksperimen | Menghindari overclaim bahwa satu output generatif menyelesaikan QA dan citation sekaligus |

Status terbaru QA final:

| Item | Status | Catatan |
| --- | --- | --- |
| Konversi `qa_bank_full` ke JSON `answer + source` | ✅ | Menghasilkan `data/pasalid/qa_bank_json_final.jsonl` dengan `285` row dari `12` laws |
| Split final QA lebih besar | ✅ | `data/pasalid/json_final_split/`: train `146`, valid `15`, test_seen `73`, test_unseen `51` |
| Training TinyLlama final | ✅ | Adapter `1000` iter dan sanity-check `400` iter sudah dievaluasi |
| Hasil `C > A` pada split final konversi | ❌ | `C` berada di bawah `A` pada seen dan unseen |
| Status split final konversi | 🟡 | Berguna sebagai negative robustness check, tetapi belum layak menggantikan JSON-large pilot sebagai setup utama |
| Native JSON QA expanded dari `doc_units` | ✅ | `data/pasalid/qa_bank_json_native_expanded.jsonl` berisi `439` row dari `137` doc units dan `17` laws |
| Split native expanded | ✅ | `data/pasalid/json_native_expanded_split/`: train `182`, valid `75`, test_seen `90`, test_unseen `92` |
| Config/wrapper native expanded TinyLlama | ✅ | `configs/pasalid_experiment_native_expanded_tinyllama.yaml`, train/eval wrapper, dan preset export sudah ditambahkan |
| Training/evaluasi TinyLlama native expanded | ✅ | Seen: `C-A +0.0551`, `D-B +0.1941`; unseen: `C-A -0.0427`, `D-B +0.1145` |
| Review native expanded `B` vs `D` | ✅ | Seen mendukung `D` pada factual/evidence/source; unseen mendukung source traceability tetapi factual/evidence masih tie atau sedikit turun |
| Catatan native expanded | 🟡 | Original split memiliki `1` train dan `4` unseen-with-context sample >2048 token |
| Clean native expanded split | ✅ | Filter `--max-source-chars 3000` menghasilkan `432` row; semua train/valid/test with-context `0` sample >2048 token |
| Config/wrapper clean TinyLlama | ✅ | `configs/pasalid_experiment_native_expanded_clean_tinyllama.yaml`, train/eval wrapper, dan preset export clean sudah ditambahkan |
| Training/evaluasi clean TinyLlama | ✅ | Seen: `C-A -0.0001`, `D-B +0.2393`; unseen: `C-A -0.0510`, `D-B +0.1025` |
| Kesimpulan clean TinyLlama | ✅ | Clean split memperkuat `D` sebagai context-use adapter; `C` tidak stabil dan tidak layak jadi klaim utama final |
| Review clean `B` vs `D` | ✅ | LLM-assisted review `20` contoh/split mendukung `D` pada factual/evidence/source baik seen maupun unseen |
| Benchmark efisiensi clean per-example | ✅ | Token dihitung dengan tokenizer; `C` mengurangi prompt token tetapi belum lebih cepat; `D` kualitas tinggi dengan latency lebih mahal dari `B` |
| Natural legal QA final targeted-completeness | ✅ | `538` row; train `340`, valid `38`, test_seen `122`, test_unseen `38`; report-like rows `0`, targeted completeness/transition rows `154` |
| Evaluasi natural legal QA final | ✅ | Seen: `D-B +0.0634`; unseen: `D-B +0.0530`; `D` lebih baik pada answer F1, citation, dan copy-rate |
| Review pairwise natural `B` vs `D` | ✅ | `30` seen + `30` unseen; overall D wins `17/30` seen dan `16/30` unseen |
| Failure audit natural `D` | ✅ | Reproducible audit: strict failure seen `18/30`, unseen `16/30`; residual dominan entity/count/list, incomplete focus, extractive output, dan source/format raw |
| Benchmark efisiensi natural QA per-example | ✅ | `20` contoh/split; `D` menambah latency atas `B`, sedangkan `C` prompt pendek tetapi paling lambat |
| Format/source constrained natural QA | ✅ | Post-processing constrained membuat valid JSON/citation `1.0`; pairwise constrained menjadi seen `B 12`, `D 11`, tie `7` dan unseen `B 11`, `D 13`, tie `6` |
| Next action QA | 🟡 | Tambah checker/validator isi jawaban untuk angka, provinsi asal, Lembaran Negara, daftar wilayah, dan status repeal; jangan hanya memaksa JSON/source |

## 8) Batas Klaim Final yang Direkomendasikan

| Jenis Klaim | Formulasi Aman |
| --- | --- |
| Internalization | Adapter LoRA menunjukkan internalisasi parsial hanya pada setup tertentu; clean split menunjukkan `C` tidak stabil dan tidak boleh menjadi klaim utama tunggal |
| Context baseline | Kondisi `B` tetap menjadi upper bound praktis terhadap adapter-only `C`; kondisi `D` diuji terpisah sebagai adapter dengan konteks |
| Context-use adaptation | Kondisi `D` menunjukkan sinyal positif pada TinyLlama clean, natural legal QA targeted-completeness, dan Mistral q4 long, tetapi gagal pada Qwen3; setelah format/source constraint diterapkan sama ke `B/D`, keunggulan `D` lebih kecil sehingga klaim harus model-dependent dan task-dependent |
| Source traceability | Source attribution belum reliabel jika dipaksa muncul sebagai bagian dari output QA generatif tunggal |
| Attribution task | Source attribution lebih menjanjikan jika diformulasikan sebagai task khusus source prediction |
| Efisiensi | Keunggulan utama `C` adalah pengurangan panjang prompt dibanding `B/D`, bukan latency; `B` paling efisien untuk retrieval-only, sedangkan `D` menukar latency tambahan dengan source discipline lebih baik. Klaim memory tetap dibatasi karena masih proxy proses |

Klaim yang sebaiknya dihindari:

| Klaim | Alasan |
| --- | --- |
| Adapter sudah menggantikan kebutuhan konteks dokumen | `B` masih konsisten lebih kuat daripada adapter-only `C` |
| Adapter dengan konteks selalu lebih baik daripada base dengan konteks | Qwen3 menunjukkan `D < B` pada seen dan unseen |
| Model sudah memberi citation legal yang reliabel | Citation metrics QA utama masih sangat lemah |
| Source-prediction menunjukkan internalisasi penuh | Test eksplisit masih berisiko mengandung jawaban sumber di prompt |
| Model lebih besar otomatis lebih baik | Qwen3 dan Mistral tidak mengungguli TinyLlama pada QA utama saat ini |

## 9) Validasi Cepat Adaptasi Hypernetwork

Bagian ini menjawab apakah arah Doc-to-LoRA-style hypernetwork dapat ditambahkan tanpa mengubah temuan eksperimen yang sudah ada.

### Kesimpulan kelayakan

| Aspek | Status | Catatan |
| --- | --- | --- |
| Kompatibel dengan pipeline LoRA existing | ✅ | Pipeline sudah bisa memuat adapter `.npz`; hypernetwork dapat menghasilkan `.npz` dengan key dan shape yang sama |
| Perlu mengubah hasil A/B/C yang sudah ada | ❌ | Tidak perlu; hypernetwork diposisikan sebagai kondisi baru `H`, bukan pengganti `A/B/C` |
| Ukuran output penuh TinyLlama LoRA | 🟡 | Adapter TinyLlama 4 layer q/v rank-8 berisi `204,800` parameter, masih kecil sebagai artifact, tetapi besar untuk MLP output naif |
| Prototype paling aman | ✅ | Mulai dari generated `lora_b` atau mixture-of-LoRA basis, bukan generate semua matrix LoRA penuh |
| Risiko utama | 🟡 | Dataset dokumen masih kecil untuk meta-training; perlu dimulai sebagai prototype feasibility, bukan klaim final utama |

### Kenapa tidak mengubah temuan saat ini

Temuan yang sudah ada tetap berlaku untuk **LoRA fine-tuning biasa**:

| Temuan existing | Status setelah hypernetwork ditambahkan |
| --- | --- |
| `B` context-based masih upper bound praktis | Tetap berlaku sebagai baseline `B` |
| `C` adapter-only tidak stabil | Tetap berlaku sebagai baseline LoRA konvensional |
| Source attribution sulit tanpa sumber eksplisit | Tetap berlaku; hypernetwork menjadi metode baru untuk diuji |
| Split konversi naratif noisy | Tetap diperlakukan sebagai negative robustness check |

Hypernetwork tidak membatalkan hasil ini karena ia menjawab pertanyaan yang berbeda: apakah adapter dapat **dihasilkan dari dokumen**, bukan apakah LoRA biasa cukup setelah fine-tuning QA.

### Kondisi D: LoRA + konteks dokumen

Kondisi `D` ditambahkan untuk menguji apakah LoRA membantu model memanfaatkan konteks dokumen, bukan menggantikan konteks.

| Model | Split | B F1 | D F1 | Selisih | Catatan |
| --- | --- | ---: | ---: | ---: | --- |
| TinyLlama | JSON-large seen | 0.4524 | 0.4670 | +0.0145 | `D > B` |
| TinyLlama | JSON-large unseen | 0.3500 | 0.3643 | +0.0143 | `D > B` |
| Qwen3 | JSON-large seen | 0.3565 | 0.3027 | -0.0538 | `D < B` |
| Qwen3 | JSON-large unseen | 0.3902 | 0.3064 | -0.0838 | `D < B` |
| Mistral q4 long | JSON-large seen | 0.3528 | 0.3992 | +0.0463 | `D > B` |
| Mistral q4 long | JSON-large unseen | 0.3936 | 0.4497 | +0.0561 | `D > B` |

Interpretasi:

| Temuan | Makna |
| --- | --- |
| `D > B` pada TinyLlama dan Mistral q4 long | LoRA feasible diposisikan sebagai context-use/domain adapter pada sebagian model |
| `D < B` pada Qwen3 | Adapter dapat mengganggu penggunaan konteks; efek D tidak universal |
| Gain positif masih kecil hingga sedang | Perlu validasi pada dataset native JSON yang lebih besar |
| Citation component masih lemah | Ada sinyal format/source adherence pada TinyLlama, tetapi belum cukup untuk traceability penuh |
| Posisi tesis | `C` tetap branch internalization; `D` menjadi branch context-use adaptation yang lebih kuat secara praktis |

Review berbantu LLM awal pada `10` contoh per model/split memberi triangulasi tambahan:

| Model | Split | Δ factual D-B | Δ evidence D-B | Δ source D-B | Catatan |
| --- | --- | ---: | ---: | ---: | --- |
| TinyLlama | seen | +0.20 | +0.20 | +0.50 | Mendukung `D` |
| TinyLlama | unseen | -0.10 | -0.10 | +0.30 | Campuran; perlu sample lebih besar |
| Qwen3 | seen | -0.40 | -0.30 | +0.40 | Counterexample kuat untuk `D` |
| Qwen3 | unseen | +0.10 | +0.10 | +0.10 | Tipis mendukung `D`, tetapi automatic F1 tetap `D < B` |
| Mistral q4 long | seen | +0.40 | +0.30 | +0.20 | Mendukung `D` |
| Mistral q4 long | unseen | +0.10 | +0.10 | +0.20 | Tipis mendukung `D` |

Kesimpulan review awal: `D` layak dipertahankan sebagai cabang context-use adaptation, tetapi klaimnya harus tetap dibatasi dan divalidasi dengan review manual final yang lebih besar.

### Kondisi eksperimen tambahan

| Kondisi | Deskripsi | Peran |
| --- | --- | --- |
| `A` | Base no-context | Lower bound |
| `B` | Base with-context | Context upper bound |
| `C` | Fine-tuned LoRA no-context | Baseline internalisasi konvensional |
| `D` | Fine-tuned LoRA with-context | Context-use adapter check |
| `H` | Hypernetwork-generated LoRA no-context | Doc-to-LoRA-inspired prototype |

Target validasi cepat bukan `H > B`, tetapi:

| Pertanyaan | Interpretasi |
| --- | --- |
| Apakah `H > A`? | Ada sinyal document-conditioned adaptation |
| Apakah `H` mendekati `C`? | Hypernetwork mulai meniru LoRA fine-tuning biasa |
| Apakah `H` gagal total? | LoRA biasa tetap baseline; Doc-to-LoRA penuh butuh desain lebih kuat |

### Prototype paling feasible

Urutan prototype yang disarankan:

| Tahap | Desain | Alasan |
| --- | --- | --- |
| 1 | Generate adapter `.npz` dummy dengan shape yang sama | Validasi integration path tanpa training hypernetwork |
| 2 | Document embedding -> MLP -> `lora_b` saja | Output jauh lebih kecil daripada full LoRA; `lora_a` bisa fixed/global |
| 3 | Document embedding -> koefisien mixture-of-LoRA basis | Paling ringan untuk meta-learning kecil |
| 4 | Full LoRA generation | Ditunda; paling mahal dan paling berisiko overfit |

Catatan dari repo SakanaAI `Doc-to-LoRA`:

| Komponen Doc-to-LoRA | Relevansi untuk repo ini |
| --- | --- |
| `ModulatedPretrainedModel.internalize(doc)` | Konsep utama: dokumen diproses sekali untuk menghasilkan LoRA yang memengaruhi generation berikutnya |
| `HyperLoRA` | Hypernetwork menghasilkan matrix LoRA per layer dan module dari fitur context encoder |
| `target_modules: [down_proj]` pada config NIAH/main | Mereka membatasi target module agar output hypernetwork lebih feasible |
| `ctx_encoder_type: per_layer_activations` dan Perceiver aggregator | Implementasi penuh cukup berat; untuk validasi cepat repo ini sebaiknya tidak langsung direplikasi penuh |
| Runtime `lora_forward` patching | Ide yang bisa ditiru; namun untuk repo ini jalur `.npz` lebih sederhana dan tidak mengganggu pipeline existing |

Untuk TinyLlama setup saat ini:

| Target generation | Perkiraan parameter output per adapter |
| --- | ---: |
| Full q/v LoRA 4 layer rank-8 | `204,800` |
| `lora_b` only q/v 4 layer rank-8 | `73,728` |
| Mixture coefficients untuk 8 basis LoRA | `8` |

### Validasi integrasi yang sudah dilakukan

| Item | Status | Catatan |
| --- | --- | --- |
| Script generator adapter prototype | ✅ | `scripts/prototype_doc_to_lora_adapter.py` menghasilkan adapter `.npz` dari doc unit memakai schema adapter TinyLlama existing |
| Adapter prototype zero | ✅ | `outputs/adapters/adapters_pasalid_hyperproto_tinyllama_zero.npz` berisi LoRA nol untuk validasi loading |
| Adapter prototype hash | ✅ | `outputs/adapters/adapters_pasalid_hyperproto_tinyllama_hash.npz` berisi LoRA deterministik berbasis hash dokumen dengan skala kecil |
| Manifest adapter | ✅ | Manifest `.json` mencatat hash dokumen, mode, jumlah tensor, dan jumlah parameter |
| Export/evaluator compatibility | ✅ | Kedua adapter bisa dimuat oleh `scripts/export_pasalid_experiment_abc.py` tanpa perubahan model core |
| Efek output prototype | ✅ | Adapter `zero` menghasilkan skor sama dengan base no-context pada smoke test, sesuai ekspektasi karena LoRA nol tidak mengubah model |
| Mixture-of-LoRA prototype | ✅ | `scripts/prototype_doc_to_lora_mixture.py` menghasilkan adapter campuran dari beberapa basis LoRA existing |
| Smoke test mixture | ✅ | Adapter mixture bisa dimuat dan dievaluasi; pada 5 contoh JSON-large seen, `H_mixture` F1 `0.2544` vs base `A` F1 `0.2406` |
| Oracle grid-search mixture | ✅ | `scripts/grid_search_lora_mixture.py` mencari koefisien global dan per-law terbaik dari basis LoRA existing |

Makna validasi:

| Pertanyaan | Jawaban |
| --- | --- |
| Apakah repo ini bisa menerima adapter yang dihasilkan secara eksternal? | Ya, selama key dan shape `.npz` mengikuti adapter LoRA existing |
| Apakah ini sudah hypernetwork terlatih? | Belum; ini baru integration prototype untuk jalur `document -> adapter artifact -> evaluation` |
| Apakah temuan A/B/C berubah? | Tidak; prototype diposisikan sebagai kondisi `H` tambahan |
| Langkah berikutnya | Ganti koefisien hash/deterministik dengan hypernetwork kecil yang belajar memilih koefisien mixture dari document embedding |

### Hasil oracle mixture awal

Basis yang dipakai:

| Basis | Adapter |
| --- | --- |
| `b0` | `adapters_pasalid_tinyllama_experiment.npz` |
| `b1` | `adapters_pasalid_tinyllama_final.npz` |
| `b2` | `adapters_pasalid_tinyllama_final_400.npz` |

Kandidat koefisien: single basis, pairwise average, dan uniform average.

| Split | Best Single/Basis F1 | Best Global Mixture F1 | Per-Law Oracle F1 | Catatan |
| --- | ---: | ---: | ---: | --- |
| JSON-large seen (`48` rows) | 0.2996 (`b0`) | 0.3259 (`0.5*b0 + 0.5*b2`) | 0.3584 | Routing per-law memberi sinyal paling kuat |
| JSON-large unseen (`21` rows) | 0.2731 (`b2`) | 0.2796 (`0.5*b1 + 0.5*b2`) | 0.2796 | Hanya satu held-out law, jadi oracle sama dengan global |

Interpretasi:

| Temuan | Makna |
| --- | --- |
| Mixture global bisa mengungguli basis tunggal | LoRA fusion bukan sekadar kosmetik; ada sinyal komplementaritas antar adapter |
| Oracle per-law lebih tinggi pada seen split | Learned router berbasis dokumen/law layak diuji |
| Citation tetap nol | Mixture saat ini membantu answer overlap, belum memperbaiki traceability |
| Unseen improvement kecil | Generalisasi routing ke held-out law masih belum cukup kuat |

### Hasil learned router awal

Router kecil dibuat di `scripts/train_lora_mixture_router.py`. Router ini memakai fitur law/doc sederhana:

| Fitur | Contoh |
| --- | --- |
| metadata hukum | nomor UU, tahun, jumlah article unit |
| panjang dokumen | rata-rata karakter doc unit |
| indikator isi | ada/tidaknya kata pidana/denda, tetap berlaku/dicabut |
| komposisi unit | rasio `article_qa` |

Router dilatih dari oracle per-law pada JSON-large seen split, lalu diuji pada seen dan unseen.

| Evaluasi | Router F1 | Best Global Mixture F1 | Oracle F1 | Catatan |
| --- | ---: | ---: | ---: | --- |
| Seen | 0.3430 | 0.3259 | 0.3584 | Router mendekati oracle dan mengungguli global mixture |
| Unseen | 0.2576 | 0.2796 | 0.2796 | Router belum generalize ke held-out law |

Interpretasi:

| Temuan | Makna |
| --- | --- |
| Router seen > global mixture | Document/law-conditioned routing memang punya sinyal manfaat |
| Router seen < oracle | Masih ada ruang untuk fitur/arsitektur router lebih baik |
| Router unseen < global mixture | Fitur law-level sederhana belum cukup untuk generalisasi ke law baru |
| Klaim aman | Prototype mendukung feasibility branch `H`, bukan klaim hypernetwork final |

### Rekomendasi scope tesis

Hypernetwork sebaiknya masuk sebagai **branch feasibility/prototype**, bukan mengganti eksperimen utama yang sudah ada.

Formulasi aman:

| Bagian tesis | Posisi |
| --- | --- |
| Eksperimen A/B/C/D | Baseline empiris LoRA konvensional |
| Eksperimen H | Prototype Doc-to-LoRA-inspired document-conditioned adapter generation |
| Klaim utama | LoRA konvensional punya batas; hypernetwork diuji sebagai arah adaptasi yang lebih dekat ke Doc-to-LoRA |
| Klaim yang dihindari | Hypernetwork sudah mereplikasi Doc-to-LoRA penuh |
