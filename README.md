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
