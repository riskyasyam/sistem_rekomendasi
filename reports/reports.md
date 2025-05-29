# Laporan Proyek Machine Learning - Muhammad Rizky Asyam Haidar

## Project Overview
![Header](header.png)

Dalam era digital saat ini, platform e-commerce telah menjadi bagian integral dari pengalaman berbelanja konsumen. Dengan jutaan produk yang tersedia, pengguna seringkali dihadapkan pada tantangan untuk menemukan item yang paling relevan dengan kebutuhan dan preferensi mereka. Kesulitan ini dapat menyebabkan penurunan keterlibatan pengguna, tingkat konversi yang lebih rendah, dan akhirnya mengurangi kepuasan pelanggan. Salah satu solusi paling efektif untuk mengatasi masalah ini adalah dengan mengimplementasikan sistem rekomendasi produk yang cerdas dan personal.

Sistem rekomendasi bertujuan untuk menyaring informasi dan menyajikan item-item yang paling mungkin diminati oleh pengguna. Dengan menganalisis data historis seperti pembelian sebelumnya, rating produk, dan perilaku penjelajahan, sistem ini dapat memprediksi preferensi pengguna dan memberikan saran produk yang dipersonalisasi. Implementasi sistem rekomendasi yang efektif dapat secara signifikan meningkatkan pengalaman pengguna, mendorong penemuan produk baru, meningkatkan penjualan, dan membangun loyalitas pelanggan.

Proyek ini berfokus pada pengembangan sistem rekomendasi produk untuk dataset e-commerce Olist, sebuah platform e-commerce besar di Brazil. Dengan memanfaatkan data transaksi, ulasan produk, dan detail pelanggan, proyek ini akan mengeksplorasi dan mengimplementasikan model Collaborative Filtering untuk menghasilkan rekomendasi produk yang relevan bagi pengguna. Diharapkan sistem ini dapat menjadi dasar untuk meningkatkan interaksi pengguna dan memberikan nilai tambah bagi platform e-commerce.

**Mengapa dan Bagaimana Masalah Tersebut Harus Diselesaikan:**
- Mengapa: E-commerce dengan katalog produk yang besar menghadapi masalah "information overload", di mana pengguna kesulitan menemukan produk yang mereka inginkan atau butuhkan. Hal ini dapat menyebabkan frustrasi, kehilangan potensi penjualan, dan penurunan loyalitas pelanggan. Sistem rekomendasi membantu mempersonalisasi pengalaman belanja, membuatnya lebih efisien dan menyenangkan bagi pengguna.
- Bagaimana: Dengan menganalisis data perilaku pengguna (seperti rating dan pembelian) dan/atau atribut produk, sistem rekomendasi dapat mengidentifikasi pola dan preferensi. Berdasarkan pola ini, sistem dapat menyarankan produk yang kemungkinan besar akan relevan bagi pengguna, baik yang sudah mereka kenal maupun produk baru yang mungkin mereka lewatkan (serendipity).

**Referensi Terkait:**
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. In Recommender Systems Handbook (pp. 1-35). Springer, Boston, MA. (Menjelaskan dasar-dasar sistem rekomendasi dan berbagai pendekatannya).
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th international conference on World Wide Web (pp. 285-295). (Detail mengenai salah satu teknik Collaborative Filtering yang populer).
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37. (Membahas teknik faktorisasi matriks seperti SVD yang sering digunakan dalam Collaborative Filtering).

## Business Understanding

### Problem Statements

- Bagaimana cara meningkatkan engagement pengguna dengan platform e-commerce Olist dengan menyajikan produk yang lebih relevan dengan preferensi individual mereka?
- Bagaimana cara membantu pengguna menemukan produk baru yang mungkin mereka sukai, bahkan jika mereka belum pernah mencarinya secara eksplisit, untuk meningkatkan variasi pembelian dan kepuasan?
- Bagaimana cara mengurangi information overload yang dialami pengguna saat menjelajahi katalog produk yang sangat besar, sehingga pengalaman berbelanja menjadi lebih efisien dan menyenangkan?

### Goals

1. Membangun model sistem rekomendasi yang mampu memberikan daftar produk yang dipersonalisasi kepada setiap pengguna berdasarkan riwayat interaksi dan rating mereka.
2. Meningkatkan kemungkinan pengguna melakukan pembelian dengan menyarankan produk yang relevan dan produk pelengkap (cross-sell).
3. Meningkatkan kepuasan dan loyalitas pelanggan dengan memberikan pengalaman belanja yang lebih personal dan membantu mereka menemukan produk yang sesuai dengan kebutuhan mereka.

    ### Solution statements
    Untuk mencapai tujuan di atas, diajukan beberapa pendekatan solusi sistem rekomendasi:
    1. Collaborative Filtering (Penyaringan Kolaboratif):
        - User-Based Collaborative Filtering (UBCF): Merekomendasikan item kepada pengguna berdasarkan item yang disukai oleh pengguna lain yang memiliki profil kemiripan (misalnya, pola rating) dengan pengguna target.
        - Item-Based Collaborative Filtering (IBCF): Merekomendasikan item yang mirip dengan item yang pernah disukai atau dibeli oleh pengguna target. Kemiripan antar item dihitung berdasarkan bagaimana pengguna lain memberi rating pada item-item tersebut secara bersamaan.
        - Model-Based Collaborative Filtering (menggunakan Faktorisasi Matriks seperti SVD): Mempelajari faktor laten dari data rating pengguna-item untuk memprediksi rating pada item yang belum pernah dilihat pengguna. Pendekatan ini dipilih karena kemampuannya menangani data yang sparse dan memberikan prediksi yang cukup akurat.
    2. Content-Based Filtering (Penyaringan Berbasis Konten):
        - Merekomendasikan item yang memiliki atribut atau fitur (misalnya, kategori produk, deskripsi) yang mirip dengan item yang pernah disukai pengguna di masa lalu. Pendekatan ini berguna untuk merekomendasikan produk baru yang belum memiliki banyak interaksi atau untuk pengguna dengan sedikit riwayat.
    3. Hybrid Approach:
        - Menggabungkan kekuatan dari Collaborative Filtering dan Content-Based Filtering untuk mengatasi kelemahan masing-masing pendekatan dan menghasilkan rekomendasi yang lebih robust dan akurat.
    Pada proyek ini, fokus utama akan diberikan pada implementasi Model-Based Collaborative Filtering menggunakan Singular Value Decomposition (SVD) dari library Surprise dan juga eksplorasi model Collaborative Filtering berbasis Neural Network menggunakan TensorFlow/Keras.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah "Brazilian E-Commerce Public Dataset by Olist" yang tersedia di Kaggle. Dataset ini berisi informasi mengenai sekitar 100.000 pesanan dari tahun 2016 hingga 2018 yang dibuat di berbagai marketplace di Brazil. Fitur-fiturnya memungkinkan untuk melihat pesanan dari berbagai dimensi: mulai dari status pesanan, harga, pembayaran, kinerja pengiriman, hingga lokasi pelanggan, atribut produk, dan ulasan yang ditulis oleh pelanggan.

Tautan : [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

**Dataset ini terdiri dari beberapa file CSV yang saling terkait:**
- olist_customers_dataset.csv
- olist_geolocation_dataset.csv
- olist_order_items_dataset.csv
- olist_order_payments_dataset.csv
- olist_order_reviews_dataset.csv
- olist_orders_dataset.csv
- olist_products_dataset.csv
- olist_sellers_dataset.csv
- product_category_name_translation.csv

**Jumlah Data Awal:**
- Jumlah data customer: (99441, 5)
- Jumlah data geolocation: (1000163, 5)
- Jumlah data order_items: (112650, 7)
- Jumlah data order_payments: (103886, 5)
- Jumlah data order_reviews: (99224, 7)
- Jumlah data orders: (99441, 8)
- Jumlah data products: (32951, 9)
- Jumlah data sellers: (3095, 4)
- Jumlah data product_category_name_translation: (71, 2)

**Kondisi Data Awal (Missing Values):**
- Setelah dilakukan inspeksi data awal menggunakan metode .info() dan .isnull().sum() pada setiap dataset, ditemukan beberapa kondisi data yang perlu diperhatikan:
    - Dataset olist_order_reviews_dataset.csv memiliki jumlah missing values yang signifikan pada kolom review_comment_title (87656 missing) dan review_comment_message (58247 missing). Hal ini wajar karena tidak semua pelanggan meninggalkan komentar detail pada ulasan mereka. Kolom review_score yang krusial untuk proyek ini tidak memiliki missing values.
    - Dataset olist_orders_dataset.csv memiliki missing values pada kolom-kolom terkait waktu pengiriman seperti order_approved_at (160 missing), order_delivered_carrier_date (1783 missing), dan order_delivered_customer_date (2965 missing). Ini mengindikasikan adanya pesanan yang mungkin belum selesai diproses atau dibatalkan.
    - Dataset olist_products_dataset.csv memiliki missing values pada beberapa kolom fitur produk. Kolom product_category_name memiliki 610 missing values, yang juga berdampak pada kolom fitur lain seperti product_name_lenght, product_description_lenght, dan product_photos_qty yang memiliki jumlah missing values yang sama. Kolom product_weight_g, product_length_cm, product_height_cm, dan product_width_cm masing-masing memiliki 2 missing values.
    - Dataset lainnya (customer, geolocation, order_items, order_payments, sellers, dan product_category_name_translation) tidak menunjukkan adanya missing values pada kolom-kolomnya.
- Penanganan missing values ini akan dilakukan pada tahap Data Preparation.

**Deskripsi Variabel Utama yang Digunakan:**
- olist_customers_dataset.csv:
    - customer_id: ID unik untuk setiap entri pelanggan terkait satu pesanan.
    - customer_unique_id: ID unik untuk setiap pelanggan (identifier utama pelanggan).
    - customer_zip_code_prefix: Lima digit pertama kode pos pelanggan.
    - customer_city: Kota pelanggan.
    - customer_state: Negara bagian pelanggan.
- olist_orders_dataset.csv:
    - order_id: ID unik untuk setiap pesanan.
    - customer_id: Foreign key ke tabel customer.
    - order_status: Status pesanan (misalnya, delivered, shipped, canceled).
    - order_purchase_timestamp: Timestamp pembelian.
    - order_approved_at: Timestamp persetujuan pembayaran.
    - order_delivered_carrier_date: Timestamp penyerahan pesanan ke pihak logistik.
    - order_delivered_customer_date: Timestamp pesanan diterima pelanggan.
    - order_estimated_delivery_date: Estimasi tanggal pengiriman.
- olist_order_items_dataset.csv:
    - order_id: Foreign key ke tabel orders.
    - order_item_id: Nomor sekuensial item dalam satu pesanan.
    - product_id: ID unik untuk setiap produk.
    - seller_id: ID unik untuk setiap penjual.
    - shipping_limit_date: Batas tanggal pengiriman oleh penjual.
    - price: Harga item.
    - freight_value: Biaya pengiriman item.
- olist_order_reviews_dataset.csv:
    - review_id: ID unik untuk setiap ulasan.
    - order_id: Foreign key ke tabel orders.
    - review_score: Skor rating yang diberikan pelanggan (1 hingga 5). Ini akan menjadi target utama untuk model Collaborative Filtering.
    (Kolom ulasan lainnya seperti review_comment_title, review_comment_message, dll.)
- olist_products_dataset.csv:
    - product_id: ID unik untuk setiap produk.
    - product_category_name: Nama kategori produk (dalam bahasa Portugis).
    - (Kolom fitur produk lainnya seperti product_name_lenght, product_description_lenght, dll.)
- product_category_name_translation.csv:
    - product_category_name: Nama kategori produk (dalam bahasa Portugis).
    - product_category_name_english: Terjemahan nama kategori produk ke bahasa Inggris.
**Visualisasi Data dan EDA (Contoh):**
- Distribusi Review Score
Visualisasi distribusi review_score menunjukkan bahwa sebagian besar pelanggan memberikan rating tinggi (4 dan 5). Skor 5 adalah yang paling dominan, diikuti oleh skor 1, kemudian skor 4, 3, dan 2. Ini mengindikasikan bahwa pelanggan cenderung memberikan ulasan yang sangat positif atau sangat negatif, dengan kecenderungan kuat ke arah positif.
![Grafik Distribusi Review Score Pelanggan](distribusi_review_score.png)
- Kategori Produk (Asli) Terpopuler Berdasarkan Jumlah Item Terjual:
Berdasarkan jumlah item yang terjual, kategori produk "cama_mesa_banho" (tempat tidur, meja, & perlengkapan mandi) adalah yang paling populer, diikuti oleh "beleza_saude" (kecantikan & kesehatan), dan "esporte_lazer" (olahraga & rekreasi). Ini memberikan gambaran mengenai jenis produk yang paling diminati oleh pelanggan olist.
![Grafik Distribusi Top 10 Produk Terpopuler](distribusi_review_score.png)

## Data Preparation
Tahapan data preparation dilakukan untuk membersihkan, mentransformasi, dan menggabungkan data dari berbagai sumber menjadi satu dataset yang siap digunakan untuk pemodelan. 
1. Menggabungkan Data Pesanan dan Pelanggan: Tabel orders digabungkan dengan customer untuk mendapatkan customer_unique_id.
- Alasan: Memastikan identifikasi pelanggan yang konsisten.
2. Menangani Missing Values Awal: dropna() dilakukan pada data gabungan untuk menghilangkan baris dengan NaN pada kolom tanggal pengiriman krusial.
- Alasan: Fokus pada pesanan yang telah diproses dengan informasi lengkap.
3. Filter Pesanan yang 'Delivered': Hanya pesanan dengan order_status == 'delivered' yang dipilih.
- Alasan: Menggunakan transaksi yang berhasil sebagai dasar rekomendasi.
4. Menggabungkan dengan Item Pesanan: Data pesanan yang sudah difilter digabungkan dengan order_items.
- Alasan: Mendapatkan detail product_id untuk setiap pesanan.
5. Menggabungkan dengan Ulasan Pesanan: Hasilnya digabungkan dengan order_reviews untuk mendapatkan review_score.
- Alasan: review_score akan digunakan sebagai rating eksplisit.
6. Membuat DataFrame Rating Eksplisit: df_explicit_ratings dibuat dengan menghilangkan baris yang review_score-nya NaN dari data interaksi.
- Alasan: Model CF berbasis rating eksplisit memerlukan data rating yang valid.
7. Menerjemahkan Nama Kategori Produk dan Menangani Missing Values: Tabel products digabungkan dengan product_category_name_translation. Missing values pada product_category_name_english kemudian diisi dengan "No Category English" menggunakan fillna().
- Alasan: Memudahkan pemahaman fitur kategori dan memastikan tidak ada NaN pada kolom ini untuk display. Ini merupakan perbaikan dari proses dropna() yang terlalu agresif sebelumnya.
8. Menggabungkan Semua Informasi Utama: df_explicit_ratings digabungkan dengan fitur produk (product_id dan product_category_name_english) untuk membentuk all_data_df. Karena NaN pada product_category_name_english sudah ditangani pada langkah sebelumnya di products_translated, kolom ini di all_data_df kini bersih dari NaN yang disebabkan oleh proses merge ini.
- Alasan: Membuat DataFrame komprehensif untuk pemodelan.
## Modeling
Dua pendekatan model Collaborative Filtering dikembangkan: menggunakan TensorFlow/Keras dan library Surprise (SVD).

1. Model Collaborative Filtering dengan TensorFlow/Keras
(Sesuai dengan Cell 20 hingga Cell 23 pada notebook yang diperbarui).
Model faktorisasi matriks berbasis neural network.
- **Persiapan Data:** df_tf dibuat, ID pengguna & produk di-encode, review_score dinormalisasi [0, 1], data di-split (80% train, 20% val).
- **Arsitektur Model (RecommenderNet):** Menggunakan layer Embedding untuk pengguna dan item, serta bias. Interaksi melalui dot product, output dengan aktivasi sigmoid.
- **Kompilasi & Pelatihan:** Optimizer Adam, loss BinaryCrossentropy, metrik RootMeanSquaredError. Dilatih selama 100 epoch dengan EarlyStopping (monitor val_loss, patience=3, restore_best_weights=True).
- **Plot Hasil Pelatihan Keras:**
    - ![Model Metrics (RMSE)](metrik_rmse.png)
    - Garis Biru (Train RMSE): Ini menunjukkan nilai Root Mean Squared Error (RMSE) pada data pelatihan (training set) di setiap epoch. Terlihat bahwa Train RMSE terus menurun seiring bertambahnya epoch. Ini wajar, karena model terus belajar untuk mencocokkan data yang digunakan untuk melatihnya.
    - Garis Oranye (Validation RMSE): Ini menunjukkan nilai RMSE pada data validasi (validation set) di setiap epoch. Data validasi adalah data yang tidak digunakan untuk melatih model, sehingga metrik ini memberikan gambaran seberapa baik model dapat menggeneralisasi pada data baru yang belum pernah dilihat sebelumnya.
        - Pada plot , Validation RMSE awalnya menurun dari epoch 0 ke epoch 1 (dari sekitar 0.36 menjadi ~0.38, ini tampaknya ada kesalahan interpretasi pada nilai awal, Validation RMSE seharusnya dimulai lebih tinggi atau sama dengan Train RMSE di epoch 0 sebelum training dimulai, atau mungkin plot dimulai setelah epoch pertama training. Berdasarkan bentuk kurva, Validation RMSE sebenarnya meningkat dari epoch 0 ke epoch 1, lalu terus meningkat.)
        - Setelah epoch 0, Validation RMSE cenderung terus meningkat. Ini adalah indikasi kuat bahwa model mulai overfitting. Artinya, model terlalu "menghafal" data training sehingga performanya pada data baru (validasi) justru memburuk.
    - Epoch: Sumbu horizontal menunjukkan jumlah epoch atau iterasi pelatihan yang telah dilalui.
    - ![Model Loss](model_loss.png)
    - Garis Biru (Train Loss): Ini menunjukkan nilai fungsi kerugian (loss function, dalam kasus Anda BinaryCrossentropy) pada data pelatihan di setiap epoch. Sama seperti Train RMSE, Train Loss juga terus menurun, menunjukkan model sedang meminimalkan kesalahan pada data training.
    - Garis Oranye (Validation Loss): Ini menunjukkan nilai fungsi kerugian pada data validasi.
        - Pada plot, Validation Loss juga menunjukkan pola yang mirip dengan Validation RMSE. Setelah epoch 0, Validation Loss terus meningkat.
        - Epoch: Sumbu horizontal menunjukkan jumlah epoch pelatihan.
- **Insight Plot:** Pelatihan dengan EarlyStopping berhenti pada epoch ke-4, dan bobot terbaik dari epoch ke-1 dikembalikan. Ini menunjukkan val_loss tidak membaik setelah epoch pertama, sehingga pelatihan dihentikan lebih awal untuk mencegah overfitting.
- **Output Rekomendasi Top-N (Keras)**:
Produk dengan rating tinggi dari histori pengguna:
| ID Produk                            | Kategori                | Rating Asli |
| :----------------------------------- | :---------------------- | :---------- |
| d04857e7b4b708ee8b8b9921163edba3     | auto                    | 5.0         |

Top 10 Rekomendasi Produk:
| ID Produk                            | Kategori                | Prediksi Rating (Asli) |
| :----------------------------------- | :---------------------- | :--------------------- |
| aca2eb7d00ea1a7b8ebd4e68314663af     | furniture_decor        | 3.33                   |
| d1c427060a0f73f6b889a5c7c61f2ac4     | computers_accessories  | 3.31                   |
| 53b36df67ebb7c41585e8d54d6772e08     | watches_gifts          | 3.31                   |
| 154e7e31ebfa092203795c972e5804a6     | health_beauty          | 3.30                   |
| 389d119b48cf3043d311335e499d9c6b     | garden_tools           | 3.30                   |
| bb50f2e236e5eea0100680137654686c     | health_beauty          | 3.28                   |
| e0cf79767c5b016251fe139915c59a26     | health_beauty          | 3.28                   |
| 3dd2a17168ec895c781a9191c1e95ad7     | computers_accessories  | 3.28                   |
| 5a848e4ab52fd5445cdc07aab1c40e48     | No Category English     | 3.26                   |
| 437c05a395e9e47f9762e677a7068ce7     | health_beauty          | 3.26                   |

2. Model Collaborative Filtering dengan Surprise (SVD)
Pendekatan kedua menggunakan algoritma Singular Value Decomposition (SVD) dari library Surprise.
Menggunakan algoritma Singular Value Decomposition (SVD).
- **Persiapan Data:** df_tf_for_surprise digunakan, Reader didefinisikan, data dimuat dan di-split (80% train, 20% test).
- **Pelatihan Model SVD:** Diinisialisasi dengan parameter standar (n_factors=100, n_epochs=20, dll.) dan dilatih.
- **Output Rekomendasi Top-N (SVD)**:
Membuat Top-N rekomendasi SVD untuk pengguna: d615a46ee39d41088222d36e46fb5c03
Fungsi get_top_n_recommendations_svd siap digunakan.

Top 10 rekomendasi produk untuk pengguna d615a46ee39d41088222d36e46fb5c03 (Model SVD):
| ID Produk                            | Kategori              | Prediksi Rating |
| :----------------------------------- | :-------------------- | :-------------- |
| c7b3cf9de7be95b3e09e7a63315685eb     | luggage_accessories  | 4.82            |
| f889fb87b505b73de10c18b93352469f     | health_beauty        | 4.79            |
| a298a105818dce6878b787e4af6cff7d     | baby                  | 4.79            |
| f8b624d4e475bb8d1bddf1b65c6a64f6     | housewares            | 4.78            |
| 6a8631b72a2f8729b91514db87e771c0     | electronics           | 4.74            |
| 425db55cb3b0f5b18a2d9964da31c3c0     | stationery            | 4.74            |
| f7f59e6186e10983a061ac7bdb3494d6     | housewares            | 4.74            |
| 7e97894cc00196a56d6ec315c68b2353     | sports_leisure       | 4.74            |
| 43b54d1fc56ff394092a3dff6be2d39f     | health_beauty        | 4.73            |
| 574597aaf385996112490308e37399ce     | housewares            | 4.72            |

**Kelebihan dan Kekurangan Pendekatan yang Dipilih**
- Collaborative Filtering (Umum):
    - Kelebihan: Tidak memerlukan fitur item, mampu menemukan rekomendasi serendipitous.
    - Kekurangan: Cold Start Problem, Data Sparsity, Popularity Bias.
- SVD (Surprise):
    - Kelebihan: Implementasi matang, relatif cepat, baik untuk data sparse.
    - Kekurangan: Kurang fleksibel dibanding neural network.
- Model Keras (Faktorisasi Matriks berbasis Neural Network):
    - Kelebihan: Fleksibilitas arsitektur, potensi menangkap interaksi non-linear.
    - Kekurangan: Lebih kompleks untuk diimplementasikan dan di-tune, butuh lebih banyak data/waktu training, pemilihan loss function dan normalisasi perlu hati-hati.

## Evaluation
Metrik evaluasi yang digunakan untuk menilai performa model dalam memprediksi rating adalah:
**Model Keras (pada data validasi, metrik pada skala rating 0-1):**
    - Pelatihan dihentikan oleh EarlyStopping pada epoch ke-4, bobot terbaik dari epoch ke-1 dikembalikan.
    - val_loss terbaik (epoch 1): sekitar 0.5768
    - val_root_mean_squared_error terbaik (epoch 1, pada skala 0-1): sekitar 0.3601
    - Konversi RMSE ke skala asli (1-5, range 4): 0.3601 * 4 ≈ 1.4404
**Model SVD (Surprise, pada data test, metrik pada skala rating asli 1-5):**

RMSE: 1.2226
MAE: 0.9568

**Analisis Hasil:**

Model SVD dari library Surprise (RMSE: 1.2226) menunjukkan performa yang lebih baik dibandingkan model Keras (estimasi RMSE skala asli: 1.4404) dengan konfigurasi saat ini. Penggunaan EarlyStopping pada model Keras mencegah overfitting yang lebih parah dengan menghentikan pelatihan saat val_loss tidak lagi menurun. Untuk SVD, MAE sebesar 0.9568 mengindikasikan rata-rata kesalahan prediksi sekitar 0.96 poin pada skala 1-5, yang merupakan hasil yang cukup baik untuk baseline. Performa model Keras yang lebih rendah bisa jadi karena penghentian dini atau memerlukan tuning hyperparameter lebih lanjut dan mungkin eksperimen dengan loss function yang berbeda (misalnya, MSE untuk regresi rating).

