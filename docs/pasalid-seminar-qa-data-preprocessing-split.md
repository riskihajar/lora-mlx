# Q&A Seminar: Data Preprocessing dan Split Anti-Leakage

Dokumen ini menyimpan jawaban siap pakai untuk pertanyaan dosen saat seminar proposal terkait preprocessing data dan strategi split agar evaluasi relevan serta tidak leakage.

## Pertanyaan 1: Bagaimana kamu melakukan data preprocessing, apa saja tahap yang akan kamu lakukan?

### Jawaban Singkat

Data preprocessing saya lakukan bertahap agar dokumen hukum mentah bisa menjadi dataset QA yang dapat dievaluasi. Tahapnya dimulai dari akuisisi dokumen, validasi sumber, pembersihan teks, pemecahan dokumen menjadi unit pasal atau bagian, pembentukan pasangan QA berbasis evidence, lalu penyusunan split eksperimen untuk kondisi `A`, `B`, dan `C`.

### Jawaban Lengkap

1. Akuisisi dokumen

Mengambil dokumen dari Pasal.id API sebagai sumber terstruktur awal. Metadata yang disimpan mencakup jenis regulasi, nomor, tahun, judul, pasal atau bagian, dan teks dokumen.

2. Validasi sumber

Melakukan verifikasi silang ke sumber resmi pemerintah atau PDF resmi. Tujuannya memastikan data dari API tidak salah, tidak terpotong, dan tetap dapat dipertanggungjawabkan sebagai dokumen hukum.

3. Pembersihan teks

Menghapus noise seperti header/footer, nomor halaman, karakter rusak OCR, spasi berlebih, dan bagian yang tidak relevan. Teks juga dinormalisasi agar formatnya konsisten.

4. Segmentasi dokumen

Memecah dokumen panjang menjadi unit yang lebih kecil, misalnya pasal, ayat, atau bagian. Ini penting agar evidence span lebih jelas dan konteks tidak terlalu panjang untuk model.

5. Pembentukan QA berbasis evidence

Menyusun pertanyaan dan jawaban dari bagian dokumen yang memiliki jawaban eksplisit. Setiap QA dilengkapi evidence span atau rujukan pasal/bagian sebagai dasar jawaban.

6. Normalisasi format data

Menyimpan data dalam format terstruktur, misalnya JSONL. Format minimal berisi `question`, `answer`, `source_doc`, `source_reference`, dan metadata regulasi.

7. Filtering kualitas

Menghapus unit yang terlalu pendek, terlalu noisy, duplikat, atau terlalu panjang untuk batas konteks model. Pada tahap ini juga dicek apakah jawaban benar-benar didukung oleh evidence.

8. Penyusunan split eksperimen

Membagi data menjadi train, validation, test seen, dan test unseen. Seen digunakan untuk menguji dokumen yang masih dalam cakupan pelatihan, sedangkan unseen digunakan untuk membaca kemampuan pada dokumen atau regulasi yang tidak dilatih langsung.

9. Pembuatan format untuk kondisi eksperimen

- Kondisi `A`: prompt hanya pertanyaan tanpa dokumen.
- Kondisi `B`: prompt berisi pertanyaan dan dokumen sumber.
- Kondisi `C`: model memakai adapter LoRA, tetapi prompt tidak membawa dokumen sumber.
- Jika ada analisis tambahan, kondisi `D`: adapter LoRA dan dokumen sumber.

### Kalimat Penutup

Jadi preprocessing saya bukan hanya membersihkan teks, tetapi mengubah dokumen hukum menjadi unit evaluasi yang jelas, berbasis evidence, dan bisa dipakai secara adil untuk membandingkan kondisi tanpa konteks, dengan konteks, dan dengan adapter LoRA.

## Pertanyaan 2: Bagaimana caramu split data agar relevan dan tidak leakage?

### Jawaban Singkat

Saya melakukan split dengan memperhatikan unit dokumen hukum, bukan sekadar membagi baris QA secara acak. Tujuannya agar pertanyaan dari dokumen yang sama tidak bocor ke evaluasi yang seharusnya menguji dokumen berbeda.

### Jawaban Lengkap

1. Kelompokkan data berdasarkan dokumen atau regulasi

Semua QA memiliki metadata seperti `law_id`, jenis regulasi, nomor, tahun, dan pasal. Split tidak dilakukan langsung per baris, tetapi terlebih dahulu dikelompokkan berdasarkan `law_id`.

2. Pisahkan train, validation, test seen, dan test unseen

- `train`: QA dari sebagian dokumen untuk melatih adapter.
- `validation`: dokumen atau QA untuk monitoring konfigurasi.
- `test_seen`: QA lain dari dokumen yang regulasinya sudah ada di train, tetapi pertanyaannya berbeda.
- `test_unseen`: QA dari regulasi yang tidak masuk train sama sekali.

3. Cegah leakage antar dokumen

Untuk `test_unseen`, seluruh `law_id` ditahan dari training. Jadi tidak ada QA dari regulasi yang sama masuk train dan test unseen.

4. Cegah duplicate leakage

Mengecek pertanyaan atau jawaban yang identik atau terlalu mirip agar tidak muncul di train dan test. Jika ada duplikat atau near-duplicate, salah satunya dihapus atau dipindahkan.

5. Pisahkan konteks dan target

Pada kondisi `A` dan `C`, prompt tidak boleh membawa `source_doc`. Pada kondisi `B`, dokumen sumber memang sengaja diberikan sebagai baseline konteks. Dengan demikian, dokumen hanya muncul pada kondisi yang memang dirancang memakai konteks.

6. Audit pertanyaan yang terlalu eksplisit

Untuk evaluasi source attribution, pertanyaan yang menyebut langsung nomor UU, tahun, atau pasal dapat membuat tugas terlalu mudah. Karena itu pertanyaan eksplisit dan implisit sebaiknya dibedakan, lalu hasilnya dilaporkan terpisah.

### Kalimat Singkat Saat Seminar

Prinsip utama saya adalah split berbasis dokumen atau regulasi, bukan random per QA. Dengan begitu, test unseen benar-benar berasal dari regulasi yang tidak masuk training, sedangkan test seen tetap dipakai untuk melihat kemampuan model pada dokumen yang sudah pernah diadaptasi tetapi dengan pertanyaan berbeda.

### Jika Dosen Menekan Soal Leakage

Saya juga memastikan kondisi no-context tidak membawa dokumen sumber di prompt. Jadi kalau adapter menjawab lebih baik pada kondisi `C`, itu bukan karena dokumen bocor di prompt, tetapi karena informasi dipelajari saat pembentukan adapter.
