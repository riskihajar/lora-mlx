# Template Review Manual Pasal.id

Dokumen ini dipakai untuk menghitung metrik manual yang belum terisi penuh di proposal, terutama:

- `evidence support rate`
- `unsupported answer rate`
- penilaian `factual correctness`
- penilaian `source traceability`

## Aturan Skor

### Factual Correctness

- `0` = salah atau absurd
- `1` = sebagian benar
- `2` = benar secara substansi

### Evidence Support

- `0` = tidak didukung dokumen
- `1` = didukung sebagian
- `2` = jelas didukung dokumen

### Source Traceability

- `0` = tidak ada atau salah total
- `1` = ada sebagian
- `2` = tepat dan dapat dilacak

## Label Error Opsional

Gunakan salah satu atau lebih bila perlu:

- `supported-correct`
- `supported-partial`
- `unsupported-answer`
- `factually-wrong`
- `source-missing`
- `source-wrong`

## Tabel Review

| No | Model | Kondisi | Split | Pertanyaan Ringkas | Gold Ringkas | Prediksi Ringkas | Factual Correctness (0/1/2) | Evidence Support (0/1/2) | Source Traceability (0/1/2) | Label Error | Catatan |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | TinyLlama | A | seen |  |  |  |  |  |  |  |  |
| 2 | TinyLlama | B | seen |  |  |  |  |  |  |  |  |
| 3 | TinyLlama | C | seen |  |  |  |  |  |  |  |  |
| 4 | TinyLlama | A | seen |  |  |  |  |  |  |  |  |
| 5 | TinyLlama | B | seen |  |  |  |  |  |  |  |  |
| 6 | TinyLlama | C | seen |  |  |  |  |  |  |  |  |

## Saran Penggunaan

- Ambil sampel yang sama untuk kondisi `A`, `B`, dan `C` agar pembacaan perbandingan tetap adil.
- Gunakan minimal `20-30` pertanyaan untuk review manual awal.
- Bila ingin membedakan fokus, buat lembar terpisah untuk:
  - QA utama
  - source prediction

## Cara Membaca Hasil Agregat

- `evidence support rate`
  - proporsi jawaban dengan skor `Evidence Support >= 1`
- `unsupported answer rate`
  - proporsi jawaban dengan skor `Evidence Support = 0`
- `factual correctness rate`
  - bisa dibaca sebagai rata-rata skor atau proporsi `>= 1`
- `source traceability rate`
  - bisa dibaca sebagai rata-rata skor atau proporsi `>= 1`
