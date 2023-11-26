**Context :**

Perusahaan telekomunikasi merupakan salah satu industri yang paling penting di dunia. Terutama pada era digital yang semakin maju, persaingan di industri telekomunikasi seluler semakin ketat. Salah satu tantangan yang dihadapi provider adalah mempertahankan pelanggan agar tidak beralih ke provider lain. Dikutip dari B2B NPS dan CX Benchmarks CustomerGauge, industri telekomunikasi memiliki tingkat retensi rata-rata sekitar 69%. Meskipun angka ini belum termasuk yang paling rendah (tingkat retensi di industri grosir hanya sekitar 44%), masih ada kesempatan untuk meningkatkan performa dalam hal mempertahankan pelanggan [(sumber)](https://customergauge.com/blog/reducing-customer-churn-in-telecommunications#5%20Reasons%20Why%20Churn%20in%20Telecoms%20Is%20So%20High3). 

Sebagai seorang data science, kita diminta untuk membuat model prediksi menggunakan machine learning untuk mengetahui apakah pelanggan akan beralih atau tidak. Perusahaan telekomunikasi dapat memanfaatkan teknologi [machine learning](https://neptune.ai/blog/how-to-implement-customer-churn-prediction#:~:text=Predicting%20churn%20is%20a%20good,to%20identify%20and%20predict%20churn.) ini untuk mengenali pelanggan yang mungkin memutuskan langganan. Dengan demikian, upaya retensi dapat diarahkan secara lebih tepat kepada pelanggan-pelanggan tersebut.

Target :

0 : berlangganan

1 : Berhenti berlangganan (churn)

**Problem Statement  :**

Perusahaan telekomunikasi yang memiliki persentase churn yang tinggi berisiko mengalami kegagalan. Oleh karena itu, perlu dilakukan upaya untuk mengurangi persentase churn tersebut. Perusahaan umumnya lebih memilih untuk mempertahankan pelanggan, karena biaya yang dibutuhkan untuk mempertahankan pelanggan lebih rendah daripada biaya untuk memperoleh pelanggan baru. Berdasarkan [penelitian](https://www.outboundengine.com/blog/customer-retention-marketing-vs-customer-acquisition-marketing/), biaya untuk memperoleh pelanggan baru dapat lima kali lebih tinggi daripada biaya untuk mempertahankan pelanggan yang sudah ada. Biaya customer acquisition cost untuk industri telekomunikasi rata-rata adalah sekitar $315 [(sumber)](https://www.revechat.com/blog/customer-acquisition-cost/) per new customer.

Untuk mempertahankan pelanggan, perusahaan telekomunikasi dapat memberikan insentif retensi seperti potongan harga, paket layanan yang menarik, prioritas pelayanan, dan lain-lain. Namun, kebijakan pemberian insentif retensi belum sepenuhnya efektif. Hal ini dikarenakan insentif retensi yang diberikan secara merata kepada seluruh pelanggan akan menjadi tidak efektif dan mengurangi potensi keuntungan, terutama jika pelanggan tersebut memang loyal dan tidak ingin berhenti berlangganan.

**Goals :**

Oleh karena itu, perusahaan telekomunikasi ingin memiliki kemampuan untuk memprediksi kemungkinan pelanggan akan berhenti berlangganan. Perusahaan juga ingin mengetahui faktor-faktor yang menyebabkan pelanggan berhenti berlangganan. Dengan mengetahui faktor-faktor tersebut, perusahaan dapat melakukan tindakan untuk mencegah pelanggan berhenti berlangganan.

**Metric Evaluation :**

Target utama kami adalah untuk mengidentifikasi pelanggan yang akan berhenti berlangganan. Oleh karena itu, kami menetapkan target sebagai berikut:

Target :

0: Pelanggan yang tidak akan berhenti berlangganan
1: Pelanggan yang akan berhenti berlangganan (churn)

**Kesalahan Tipe 1 (False Positive)**: Pelanggan yang aktualnya tidak akan berhenti berlangganan tetapi diprediksi akan berhenti berlangganan. Konsekuensinya adalah pemberian insentif retensi yang tidak efektif.

**Kesalahan Tipe 2 (False Negative)**: Pelanggan yang aktualnya akan berhenti berlangganan tetapi diprediksi tidak akan berhenti berlangganan. Konsekuensinya adalah kehilangan pelanggan.

Untuk memberikan gambaran yang lebih jelas tentang konsekuensinya, kita akan mencoba menghitung dampak biayanya dengan menggunakan asumsi-asumsi berikut:
1. CLTV for churn customer = 1 / churn rate, sedangkan churn rate untuk industri telekomunikasi berada di range [20% - 40%](https://www.mdpi.com/2076-3417/11/11/4742#B1-applsci-11-04742), jadi CLTV untuk pelanggan yang churn sekitar 1/20% atau 5 bulan
2. Customer Acquisition Cost (CAC) = [315$](https://www.revechat.com/blog/customer-acquisition-cost/) / 5 = $63 perbulan untuk tiap customernya
3. Customer Retention Cost (CRC) = [1/5](https://www.outboundengine.com/blog/customer-retention-marketing-vs-customer-acquisition-marketing/) * CAC = 1/5 * $63 --> $12.6 perbulan untuk tiap customernya
4. Average Customer MonthlyCharge = $64.88 per bulan untuk tiap customernya (diambil dari statistik deskriptif dataset)


Dengan mengambil asumsi diatas sebagai dasar, kita dapat mencoba mengukur konsekuensinya seperti berikut:
1. Pemberian insentif retensi yang tidak efektif dapat mengakibatkan pemborosan biaya CRC sebesar $12.6 perbulan untuk tiap customer
2. Ketika pelanggan pergi, itu berarti pendapatan turun dan kita harus mengeluarkan biaya lagi untuk akuisisi pelanggan baru, sehingga secara keseluruhan kita mengalami kerugian sebesar $63 + $64.88 = $127.88

Berdasarkan pertimbangan konsekuensinya, maka kita akan berusaha untuk membuat model yang dapat mengurangi jumlah pelanggan yang beralih dari perusahaan tersebut, khususnya pelanggan yang sebenarnya akan beralih tetapi diprediksi tidak akan beralih (False Negative). Selain itu, model tersebut juga harus dapat meminimalisir pemberian insentif yang tidak tepat. Oleh karena itu, metric utama yang akan kita gunakan adalah f2_score, karena kita menganggap recall dua kali lebih penting daripada precision.


## *Conclusion*

- Metric utama yang akan kita gunakan adalah f2_score, karena recall kita anggap dua kali lebih penting daripada precision.

- Berdasarkan pemodelan Logistic Regression, fitur/kolom **Contract** adalah yang paling penting dan berpengaruh terhadap target (Churn), kemudian diikuti dengan **PaperlessBilling**, **InternetService** dan **MonthlyCharges**.

- Berdasarkan contoh perhitungan biaya :
  
  - Potensi kerugian yang mungkin didapat tanpa adanya penerapan machine learning diperkirakan sebesar : $30,287.68 per bulan untuk 922 pelanggan

  - Potensi kerugian yang mungkin didapat dengan menerapkan model Logistic Regression yang telah dibuat diperkirakan sebesar : $26,309.08 per bulan untuk 922 pelanggan

  - Potensi kerugian yang mungkin didapat dengan menerapkan model Adaboost Classifier yang telah dibuat diperkirakan sebesar : $26,181.2 per bulan untuk 922 pelanggan.

- Berdasarkan contoh hitungan tersebut, terlihat bahwa dengan menggunakan model kita, maka perusahaan dapat menghemat sebesar :

  - Dengan Model Decision Tree : $3978.59 per bulan untuk 922 pelanggan.

  - Dengan Model Adaboost Classifier : $4106.48 per bulan untuk 922 pelanggan.

  - Memperhitungkan bahwa jumlah pelanggan pada penyedia layanan telekomunikasi dapat mencapai jutaan, tentunya potensi penghematan akan semakin signifikan jika karakteristik pelanggan masih mencakup rentang data yang digunakan dalam proses pemodelan.


**Model Limitation**

Model ini hanya berlaku pada rentang data yang digunakan pada pemodelan ini yaitu :

- tenure antara 0 sampai dengan 72 bulan
- MonthlyCharges antara 18.8 sampai dengan 118.65
- Contract dalam jangka Month-to-month, One year, dan Two Year
- InternetService berupa 'DSL', 'Fiber Optic' dan 'No'
- Dependent, Paperless Billing dengan nilai 'Yes' atau 'No'
- OnlineSecurity, OnlineBackup, DeviceProtection, dan TechSupport berisi pilihan 'Yes', 'No' atau 'No internet service'.

Pada kasus ini, analisis dan hasil prediksi dari model yang telah dibuat tidak valid untuk :

- tenure lebih besar dari 72 bulan
- MonthlyCharges kurang dari 18.8 atau lebih besar dari 118.65
- Jenis Contract selain Month-to-month, One year, dan Two Year
- InternetService selain 'DSL', 'Fiber Optic' dan 'No'
- Dependent, Paperless Billing dengan nilai selain 'Yes' atau 'No'
- OnlineSecurity, OnlineBackup, DeviceProtection, dan TechSupport berisi pilihan selain 'Yes', 'No' atau 'No internet service'.

## *Recommendation*

Beberapa tindakan yang dapat diambil oleh perusahaan untuk mengurangi jumlah pelanggan yang akan berhenti berlangganan (churn) meliputi:
- Menawarkan insentif atau penghargaan yang menarik bagi pelanggan untuk beralih dari kontrak bulanan ke kontrak tahunan atau dua tahunan.
- Membuat program loyalitas pelanggan untuk mendorong pelanggan agar tetap berlangganan dan memiliki masa berlangganan yang lama. Program ini dapat berupa pemberian penghargaan yang besarannya disesuaikan dengan masa berlangganan. Semakin lama masa berlangganan, semakin besar penghargaan yang bisa didapat, sehingga mendorong pelanggan untuk memiliki masa berlangganan yang lebih lama.
- Melakukan survei kepuasan pelanggan secara berkala untuk mengetahui kualitas layanan yang telah diberikan dan melakukan perbaikan jika ada umpan balik negatif.


Hal-hal yang bisa dilakukan untuk mengembangkan project dan modelnya lebih baik lagi diantaranya: 

- Menambahkan fitur-fitur atau kolom baru yang dapat mengukur kepuasan pelanggan terhadap masing-masing layanan. Hal ini dapat membantu untuk mengidentifikasi apakah churn disebabkan oleh kualitas layanan yang buruk.
- Menambahkan fitur-fitur atau kolom baru yang dapat memberikan gambaran tentang penggunaan produk-produk yang ada oleh pelanggan. Dengan demikian, perusahaan dapat melakukan segmentasi pelanggan untuk menentukan produk yang paling sesuai untuk ditawarkan.
- Menambahkan data untuk kelas minoritas (Churn). Hal ini dapat membantu untuk meningkatkan performa model karena model akan memiliki lebih banyak data untuk dipelajari.
- Mencoba algoritma ML dan teknik tuning parameter yang berbeda, serta menggunakan teknik oversampling yang berbeda. Dengan demikian, perusahaan dapat menemukan kombinasi yang paling optimal untuk meningkatkan performa model.
- Menganalisis data-data yang salah diprediksi oleh model. Hal ini dapat membantu untuk mengidentifikasi alasan dan karakteristik data yang salah diprediksi.
