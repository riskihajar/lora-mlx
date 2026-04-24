# Review Outline Slide Seminar Proposal

File yang direview: `/Users/riskihajar/kuliah/tesis/lora/proposal/slides/seminar-proposal/outline-slide-proposal-1-20.md`

## Findings

### High: Slide 18 terlalu overclaim terhadap `C` adapter-only

Slide 18 menyatakan output utama adalah adapter membuat model menjawab tanpa dokumen sumber, dan `C` diharapkan jelas lebih baik dari `A` serta mendekati `B`.

Ini sesuai proposal yang sudah submit, tetapi berisiko saat seminar karena hasil eksperimen terbaru menunjukkan `C` tidak stabil.

Saran:

- Ubah narasi lisan menjadi “target yang diuji”, bukan “hasil yang diharapkan pasti”.
- Tambahkan kalimat defensif:

> Jika `C` belum stabil, hasil tersebut tetap menjadi kontribusi untuk membaca batas internalisasi adapter.

### High: Slide belum memberi ruang untuk kondisi `D`

Slide 16 hanya memuat `A/B/C`. Untuk seminar proposal, ini aman karena proposal memang sudah submit dengan tiga kondisi inti.

Namun, perlu siap menjawab jika penguji bertanya: “bagaimana kalau adapter tanpa konteks gagal?”

Saran:

- Jangan ubah slide inti jika ingin tetap konsisten dengan proposal submit.
- Tambahkan di speaker notes:

> Kondisi tambahan seperti LoRA + konteks dapat dipakai sebagai analisis lanjutan untuk membaca apakah LoRA lebih berperan sebagai context-use adapter.

### Medium: Slide 5 rumusan masalah agak melebar

Slide 5 memuat AI berdaulat, LoRA, performa, panjang dokumen, dan kelayakan. Secara isi benar, tetapi untuk slide bisa terasa seperti daftar luas.

Saran:

Jadikan pertanyaan utama satu kalimat besar:

> Apakah adapter LoRA dapat mengurangi ketergantungan pada konteks dokumen panjang tanpa mengorbankan kualitas dan evidence support?

Lalu pertanyaan lain diposisikan sebagai sub-question.

### Medium: Slide 12 target dataset bisa diserang

Slide 12 menyebut 60 dokumen dan 6 QA per dokumen. Ini sesuai proposal, tetapi saat seminar jangan terdengar sebagai dataset final yang sudah tersedia.

Jawaban lisan yang aman:

> 60 dokumen adalah target awal kurasi; final size dikunci setelah validasi evidence span.

### Medium: Slide 16 belum menyebut batas klaim `C`

Slide 16 bagus untuk pembandingan, tetapi perlu narasi bahwa keberhasilan tesis tidak bergantung pada `C` harus mengalahkan `B`.

Speaker note yang disarankan:

> `B` adalah upper-bound praktis, sedangkan `C` diuji apakah dapat mendekati `B` dengan token konteks lebih rendah.

### Medium: Slide 17 metrik evaluasi perlu konsistensi istilah

Slide 17 mencampur istilah “evidence attribution” dengan “source traceability”. Ini bisa membingungkan.

Saran istilah:

- kualitas jawaban: EM/F1
- faktualitas: factual correctness
- evidence support: evidence support rate, unsupported answer rate
- source traceability: ketepatan rujukan pasal/bagian
- efisiensi: prompt tokens, latency p50/p95, memory

### Low: Slide 10 dan 11 agak repetitif

Slide 10 dan 11 sama-sama mengulang tiga klaster literatur.

Saran:

- Slide 10 fokus pada literature map.
- Slide 11 fokus pada research position atau novelty.

### Low: Slide 14 terlalu kosong

Slide 14 hanya berisi daftar enam tahap DSRM.

Saran:

Tambahkan mapping singkat:

- Design & development = adapter + dataset + protocol
- Evaluation = A/B/C comparison
- Communication = laporan tesis dan rekomendasi desain

## Masukan Strategis

- Jangan bongkar deck besar-besaran sebelum seminar.
- Outline ini sudah aman dan konsisten dengan proposal submit.
- Yang perlu diperkuat adalah speaker notes, bukan semua slide.
- Poin paling penting yang harus diucapkan:

> Saya menguji klaim internalisasi, bukan menjamin adapter menggantikan konteks.

Jika penguji bertanya soal hasil awal, jawaban aman:

> Eksperimen pendahuluan menunjukkan adapter-only perlu dibaca defensif; justru itu memperkuat pentingnya evaluasi komparatif dan trade-off.

## Kalimat Aman Untuk Slide 18

Revisi ringan atau gunakan secara lisan:

> Hasil yang diharapkan bukan bahwa adapter selalu mengalahkan konteks dokumen, tetapi bahwa adapter dapat memberi nilai tambah terukur dibanding no-context baseline, mendekati context baseline pada kondisi tertentu, atau menunjukkan batas empiris kapan konteks eksplisit tetap diperlukan.

## Kesimpulan

Outline ini layak untuk seminar proposal.

Risiko utamanya bukan struktur slide, melainkan ekspektasi yang terlalu tinggi terhadap `C`. Dengan narasi defensif, deck ini bisa dipertahankan.
