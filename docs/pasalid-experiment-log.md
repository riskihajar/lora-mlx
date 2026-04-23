# Log Eksperimen Pasal.id

Dokumen ini menyimpan log eksperimen yang lebih detail daripada `docs/pasalid-thesis-experiment-report.md`.

Tujuannya:

- menjaga histori eksperimen tetap utuh
- mencatat format, model, split, dan metrik penting
- memisahkan log eksperimen dari narasi ringkas yang lebih siap untuk tesis

## Ringkasan Posisi Saat Ini

- status keseluruhan: pilot / preliminary experiment
- temuan paling stabil: `B > C > A` pada beberapa setup kualitas jawaban
- bottleneck utama: source traceability
- format jawaban paling menjanjikan sejauh ini: JSON `answer + source`

## Keterangan Kondisi

- `A`: base model tanpa konteks dokumen
- `B`: base model dengan konteks dokumen
- `C`: base model + adapter LoRA tanpa konteks dokumen

## Keterangan Metrik

- `F1`: metrik utama kualitas jawaban
- `EM`: metrik ketat tambahan
- `Citation EM`: exact match untuk citation
- `Citation Component Score`: kecocokan komponen citation

## Tabel Eksperimen Utama

| ID | Format | Model | Split | Ukuran | A F1 | B F1 | C F1 | Citation C | Catatan |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E1 | Naratif awal | TinyLlama | seen | 20 | 0.2596 | 0.3592 | 0.3033 | N/A | sinyal awal internalization cukup jelas |
| E2 | Legal-aware subset | TinyLlama | seen | 10 | 0.3003 | 0.4058 | 0.3577 | 0.0000 | answer quality naik, citation tetap nol |
| E3 | Refined QA style | TinyLlama | seen | 10 | 0.2569 | 0.3840 | 0.2663 | 0.0000 | tightening prompt belum memperbaiki traceability |
| E4 | Two-line `Jawaban+Sumber` | TinyLlama | seen | 10 | 0.3354 | 0.3637 | 0.3246 | 0.0000 | answer quality tetap usable |
| E5 | JSON `answer+source` | TinyLlama | seen | 10 | 0.3674 | 0.5026 | 0.3079 | 0.0250 | ada sinyal kecil pada citation component |
| E6 | JSON `answer+source` | TinyLlama | seen | 20 | 0.3538 | 0.5196 | 0.3214 | 0.0125 | sinyal citation kecil masih muncul |
| E7 | JSON `answer+source` large | TinyLlama | seen | 20 | 0.3151 | 0.5078 | 0.3439 | 0.0000 | ordering stabil `B > C > A` |
| E8 | JSON `answer+source` | Mistral q4 | seen | 10 | 0.2283 | 0.3898 | 0.2622 | 0.0000 | lebih lemah dari TinyLlama |
| E9 | JSON `answer+source` | Qwen3 | seen | 20 | 0.2041 | 0.3773 | 0.2400 | 0.0000 | belum menggeser TinyLlama |
| E10 | Source components | TinyLlama | seen | 10 | 0.3275 | 0.4839 | 0.2949 | 0.0000 | ablation negatif untuk traceability |

## Eksperimen Unseen yang Penting

### TinyLlama unseen smoke

| Format | Ukuran | A F1 | B F1 | C F1 | Catatan |
| --- | ---: | ---: | ---: | ---: | --- |
| Naratif / awal | 10 | 0.2599 | 0.2917 | 0.2412 | adapter tidak mengungguli baseline no-context |

### Mistral unseen smoke

| Format | Ukuran | A F1 | B F1 | C F1 | Catatan |
| --- | ---: | ---: | ---: | ---: | --- |
| Naratif / awal | 10 | 0.0885 | 0.2684 | 0.1754 | context membantu, adapter improve tapi tetap jauh dari B |

### Qwen3 unseen

- beberapa run attempted
- belum stabil karena evaluasi unseen masih berat dan sering timeout pada window command saat ini

## Format Ablation Summary

### 1. Free-form / naratif

- mudah dijalankan
- sulit dievaluasi ketat
- source traceability sangat lemah

### 2. Dua baris `Jawaban + Sumber`

- lebih mudah dibaca manusia
- belum cukup untuk membuat model patuh ke format sumber

### 3. JSON `answer + source`

- format paling menjanjikan sejauh ini
- answer quality tetap baik
- sempat memberi sinyal kecil citation movement
- paling cocok untuk eksperimen lanjutan

### 4. Source components

- lebih ketat secara struktur
- tetapi tidak memperbaiki citation metrics
- diperlakukan sebagai ablation negatif

## Model Summary

### TinyLlama

- model yang paling konsisten dan paling berguna sebagai baseline utama saat ini
- memberikan sinyal internalization paling jelas pada beberapa setup

### Mistral q4

- context tetap membantu kuat
- adapter improve, tetapi hasilnya belum menyalip TinyLlama
- longer run membantu sedikit, tapi tidak mengubah ranking umum

### Qwen3

- promising pada beberapa run seen
- belum menggeser TinyLlama pada setup Pasal.id ini
- unseen evaluation lebih berat secara operasional

## Temuan yang Stabil

- `B` hampir selalu kondisi terbaik
- `C` dalam beberapa setup mengungguli `A`
- kualitas jawaban dapat meningkat tanpa konteks penuh melalui adapter
- source traceability tetap belum stabil di seluruh model dan format

## Pola Error yang Penting

### Unsupported atau factually wrong answer

Contoh khas:

- pertanyaan tentang definisi wilayah atau entitas hukum dijawab dengan kalimat absurd seperti:
  - `Kabupaten Merangin adalah bahaya yang diatur dalam Pasal 1`

Makna error ini:

- model masih bisa mempertahankan sebagian pola permukaan seperti `X adalah ... dalam Pasal ...`
- tetapi gagal menjaga isi faktual yang benar
- ini menunjukkan semantic corruption, bukan sekadar variasi wording

Implikasi evaluasi:

- contoh seperti ini tidak boleh dibaca sebagai sekadar `EM = 0`
- contoh ini harus diperlakukan sebagai `unsupported answer` atau `factually wrong answer`
- kasus seperti ini memperkuat alasan mengapa metrik lexical saja tidak cukup untuk eksperimen ini

### Kategori error kerja

- `supported-correct`
  - jawaban benar secara substansi dan didukung dokumen
- `supported-partial`
  - jawaban sebagian benar tetapi tidak lengkap atau kurang presisi
- `unsupported-answer`
  - jawaban tampak menjawab, tetapi tidak didukung oleh dokumen sumber
- `factually-wrong`
  - jawaban bertentangan dengan isi dokumen atau mengandung isi absurd
- `source-missing`
  - jawaban substantif ada, tetapi attribution sumber tidak muncul atau tidak bisa diparse
- `source-wrong`
  - jawaban mungkin cukup baik, tetapi sumber yang dirujuk salah

### Rubrik manual sederhana

Untuk review manual, setiap jawaban dapat diberi tiga label ringkas:

- `factual_correctness`
  - `0` = salah atau absurd
  - `1` = sebagian benar
  - `2` = benar secara substansi
- `evidence_support`
  - `0` = tidak didukung dokumen
  - `1` = didukung sebagian
  - `2` = jelas didukung dokumen
- `source_traceability`
  - `0` = tidak ada atau salah total
  - `1` = ada sebagian
  - `2` = tepat dan dapat dilacak

## Metrik yang Masih Belum Bergerak

- `EM` hampir selalu `0`
- `Citation EM` hampir selalu `0`
- `Citation Component Score` masih sangat kecil atau kembali `0`

Makna praktisnya:

- eksperimen ini lebih kuat untuk mendukung klaim internalization pada kualitas jawaban
- eksperimen ini belum cukup kuat untuk mendukung klaim source attribution yang reliabel

## Kesimpulan Kerja Saat Ini

- status eksperimen masih paling tepat disebut pilot / preliminary
- pipeline data, format, split, model comparison, dan evaluator sudah tervalidasi
- langkah lanjutan yang paling rasional adalah memakai JSON `answer + source` sebagai format utama dan memisahkan masalah source attribution sebagai bottleneck riset tersendiri
