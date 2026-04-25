# Evaluasi Proposal Seminar Tesis

## Pendapat Utama

Proposal sudah layak dan defensible untuk seminar proposal, tetapi ada satu risiko besar: proposal masih sangat kuat mengarah ke **adapter-only internalization** atau kondisi `C`, sedangkan hasil eksperimen terbaru justru menunjukkan arah paling feasible adalah **LoRA sebagai context-use adapter** atau kondisi `D`.

Dengan demikian, riset tetap feasible, tetapi framing saat seminar harus hati-hati agar tidak overclaim bahwa adapter LoRA pasti dapat menggantikan konteks dokumen.

## Kekuatan Proposal

- Framing **AI berdaulat -> open-weight -> LoRA -> QA dokumen panjang** sudah kuat.
- Novelty defensif: bukan algoritma baru, tetapi **fokus kontribusi + desain pembandingan + evaluasi empiris**.
- DSRM cocok karena penelitian membangun artefak, dataset, protokol evaluasi, dan eksperimen.
- Metrik sudah matang: kualitas jawaban, faktualitas, evidence support, traceability, dan efisiensi.
- Cheat sheet seminar sudah siap dipakai dan narasinya konsisten.

## Risiko Utama

### 1. Klaim `C` terlalu optimistis

Proposal menargetkan adapter tanpa konteks dapat mendekati `B`. Hasil eksperimen terbaru menunjukkan bahwa `C` tidak stabil, terutama pada split unseen.

Jawaban aman saat seminar:

> Saya tidak mengunci keberhasilan pada kondisi `C` harus mengalahkan `B`; kontribusi tetap ada pada pembacaan trade-off dan batas kemampuan adapter.

### 2. Proposal belum memasukkan `D`

Hasil terbaik saat ini adalah `D = LoRA + context`, sedangkan proposal utama hanya memuat `A/B/C`.

Saat seminar, jangan langsung mengubah proposal menjadi `D` sebagai pusat. Posisikan `D` sebagai **analisis tambahan jika eksperimen utama menunjukkan adapter-only belum cukup stabil**.

### 3. Episode per dokumen vs eksperimen saat ini

Proposal bicara tentang **episode per dokumen** dan adapter dari dokumen target. Eksperimen repo saat ini lebih dekat ke **corpus/domain adapter**, bukan adapter per dokumen penuh.

Ini perlu dirapikan setelah seminar: apakah tesis akan benar-benar mengimplementasikan per-document episode, atau direvisi menjadi domain/document-corpus adapter.

### 4. Model utama

Proposal menyebut Qwen3.5-4B sebagai model utama. Eksperimen terbaru justru menunjukkan TinyLlama clean paling kuat pada setup saat ini.

Saat seminar, tetap jawab sesuai proposal: Qwen dipilih karena open-weight dan realistis. TinyLlama/Mistral bisa disebut sebagai baseline eksplorasi, bukan pengganti utama kecuali nanti direvisi pasca-seminar.

### 5. Dataset

Proposal menargetkan 60 dokumen dari UU/PP/Perpres/Permen. Eksperimen saat ini masih lebih sempit.

Jawaban aman:

> Angka 60 adalah target kurasi awal; final size akan dikunci setelah validasi kualitas dan evidence span.

## Strategi Seminar

Narasi yang disarankan:

> Proposal ini menguji apakah internalisasi dokumen ke adapter LoRA dapat mengurangi ketergantungan pada konteks panjang. Saya tidak mengklaim adapter pasti menggantikan konteks dokumen. Jika hasil menunjukkan adapter-only belum stabil, itu tetap menjadi kontribusi karena penelitian ini membaca batas kemampuan adapter dan trade-off terhadap pendekatan berbasis konteks.

Jika penguji bertanya “kalau `C` gagal?”, jawaban aman:

> Itu tetap hasil ilmiah yang valid. Penelitian ini tidak hanya mencari konfigurasi yang menang, tetapi mengevaluasi kapan adapter membantu, kapan konteks eksplisit tetap dibutuhkan, dan bagaimana trade-off kualitas, traceability, dan efisiensi.

## Saran Posisi Saat Seminar

Untuk seminar proposal, jangan ubah framing besar. Proposal sudah submit, jadi pertahankan `A/B/C` sebagai eksperimen inti.

Namun, siapkan ruang narasi bahwa eksperimen dapat diperluas dengan kondisi tambahan jika hasil awal menunjukkan kebutuhan tersebut.

## Arah Pasca-Seminar

Setelah seminar, arah tesis sebaiknya direvisi halus menjadi:

| Kondisi | Peran |
| --- | --- |
| `A` | Base no-context |
| `B` | Base + context |
| `C` | LoRA no-context untuk menguji internalisasi |
| `D` | LoRA + context untuk menguji context-use adaptation |
| `H` | Doc-to-LoRA-inspired prototype sebagai feasibility branch |

## Kesimpulan

Proposal layak maju seminar, tetapi perlu defensif pada klaim adapter-only.

Kekuatan tesis sekarang bukan bahwa `C` pasti menggantikan konteks, melainkan bahwa eksperimen dapat memetakan batas `C` dan menunjukkan bahwa `D` lebih feasible sebagai desain legal QA berbasis open-weight.

Formulasi klaim paling aman:

> Fine-tuning LoRA pada korpus hukum tidak cukup stabil untuk menggantikan konteks dokumen, tetapi dapat meningkatkan kemampuan model dalam menggunakan konteks dokumen pada inferensi legal QA. Pendekatan ini memberi trade-off antara kualitas jawaban, traceability, dan biaya inferensi, serta membuka arah document-conditioned adapter generation seperti Doc-to-LoRA.
