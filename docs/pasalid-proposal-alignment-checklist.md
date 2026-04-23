# ✅ Checklist Kesesuaian Eksperimen dengan Target Proposal

Dokumen ini memetakan target proposal terhadap status eksperimen yang sudah dibangun di repo `lora-mlx`.

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
| Evidence attribution | 🟡 | Pada QA utama masih lemah; pada source prediction sudah mulai bergerak jelas | Belum ada evaluasi manual formal yang cukup besar | Tambah review manual terstruktur |
| Evidence support rate | ❌ | Rubrik dan kategori error sudah ada, tapi belum dihitung sebagai rate | Belum ada agregasi manual final | Buat tabel manual review dan hitung proporsinya |
| Unsupported answer rate | ❌ | Kategori `unsupported-answer` dan `factually-wrong` sudah ada; heuristic flags sudah tersedia | Belum dihitung sebagai rate final | Coding manual pada subset evaluasi |
| Ketepatan rujukan pasal atau bagian dokumen | 🟡 | QA utama masih lemah pada citation metrics; source prediction branch sudah bermakna | Citation di QA utama belum stabil | Pertahankan source branch sebagai pembanding attribution |
| Jumlah token konteks saat inferensi | ❌ | Secara konsep jelas `B` > `C`, tapi belum ada pengukuran formal | Belum ada tabel token count | Hitung prompt token A/B/C |
| Latensi p50 / p95 | ❌ | Belum diukur formal | Belum ada benchmark inferensi | Buat benchmark inferensi A/B/C |
| Penggunaan memori | ❌ | Belum diukur sistematis | Definisi metrik memori belum dikunci | Tetapkan metrik memori yang konsisten |

## 3) Kesesuaian terhadap Tujuan Khusus Proposal

| Tujuan Proposal | Status | Bukti Saat Ini | Gap Utama | Next Action |
| --- | --- | --- | --- | --- |
| Merancang artefak internalisasi dokumen ke adapter LoRA | ✅ | Pipeline artefak sudah ada: ingestion, doc units, QA generation, split A/B/C, training, evaluation, reporting | - | Pertahankan dan dokumentasikan |
| Menetapkan rancangan dataset eksperimen yang terukur dan dapat direplikasi | 🟡 | Dataset pipeline sudah reproducible; split dan format sudah ada; beberapa format output sudah diuji | Skala dataset belum final; coverage regulasi belum penuh | Perluas corpus dan kunci setup final |
| Membandingkan performa A/B/C pada metrik utama dan pendukung | 🟡 | A/B/C sudah dibandingkan di beberapa model dan format; kualitas jawaban sudah cukup terisi | Akuntabilitas dan efisiensi belum terukur penuh | Lengkapi evaluasi manual dan benchmark efisiensi |
| Menganalisis sejauh mana C mendekati B dan melampaui A | 🟡 | Pola `B > C > A` muncul di beberapa setup yang stabil | Belum ada satu eksperimen final sebagai basis klaim utama | Kunci satu protokol final |
| Merumuskan rekomendasi desain sistem yang seimbang antara kualitas, traceability, dan efisiensi | 🟡 | Sudah ada rekomendasi awal: QA utama dan source attribution dipisah; JSON `answer + source` paling layak; source prediction jadi branch attribution | Aspek efisiensi belum cukup terukur | Tambah metrik efisiensi dan finalisasi rekomendasi |

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
| Evidence support rate | Belum dihitung sistematis |
| Unsupported answer rate | Belum dihitung sistematis |
| Latency p50/p95 | Belum diukur formal |
| Penggunaan memori | Belum diukur formal |
| Validasi final pada setup utama yang dikunci | Belum ada satu eksperimen final utama |

## 5) Prioritas Lanjutan 🚀

| Prioritas | Pekerjaan | Alasan |
| --- | --- | --- |
| 1 | Ukur metrik efisiensi (`token`, `latency p50/p95`, `memory`) | Ini gap utama proposal yang masih kosong |
| 2 | Lakukan manual review terstruktur untuk `evidence support rate` dan `unsupported answer rate` | Ini akan mengisi aspek akuntabilitas yang belum final |
| 3 | Kunci satu setup eksperimen final untuk QA utama | Agar klaim utama tesis punya basis yang stabil |
| 4 | Kunci satu setup eksperimen final untuk source prediction | Agar branch attribution juga final dan setara |
| 5 | Susun bab hasil dan pembahasan berdasarkan dua cabang eksperimen tersebut | Ini akan mengubah hasil pilot menjadi narasi tesis yang utuh |
