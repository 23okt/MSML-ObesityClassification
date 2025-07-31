# 🧠 Obesity Classification ML Pipeline

Proyek ini membangun sistem **klasifikasi obesitas** berbasis machine learning menggunakan **Random Forest**, dilengkapi dengan **MLflow untuk tracking eksperimen**, serta **Grafana + Prometheus** untuk observabilitas dan monitoring metrik model. Pipeline ini juga sudah terintegrasi dengan **CI/CD di GitHub**.

---

---

## 🎯 Tujuan

- Mengklasifikasikan **kategori obesitas** seseorang berdasarkan data perilaku dan fisiologis.
- Melacak, membandingkan, dan menyimpan eksperimen model secara otomatis menggunakan **MLflow**.
- Melihat metrik performa model (akurasinya) secara real-time via **Grafana** dan **Prometheus**.
- Memanfaatkan **CI/CD GitHub Actions** untuk memastikan reproducibility dan konsistensi pipeline.

---

## ⚙️ Cara Kerja Pipeline

1. **Modeling**

   - Menggunakan `RandomForestClassifier` dari scikit-learn.
   - Tracking dilakukan otomatis via `mlflow.sklearn.autolog()`.
   - Metrik utama: `accuracy`.

2. **MLflow Tracking**

   - Tracking URI: `http://127.0.0.1:5000/`
   - Menyimpan:
     - Parameter model (estimators, max depth)
     - Akurasi model
     - Serialized model (pickle)
     - Input example

3. **Monitoring: Prometheus & Grafana**

   - Prometheus scrape metrik dari aplikasi/MLflow server
   - Grafana menampilkan metrik dalam dashboard interaktif
   - Contoh tampilan:

     ![Grafana Dashboard Screenshot](grafana/dashboard.png)

4. **CI/CD dengan GitHub Actions**

   - Otomatisasi:
     - Install dependencies
     - Validasi skrip `modelling.py`
     - Jalankan pelatihan model dan cek keberhasilan logging
   - Contoh konfigurasi di `.github/workflows/ci.yml`

---

## 🚀 Menjalankan Proyek

### 1. Instalasi

```bash
git clone https://github.com/username/obesity-mlflow-pipeline.git
cd obesity-mlflow-pipeline
pip install -r requirements.txt
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
python modelling.py

🛠️ Teknologi yang Digunakan
Tools	Keterangan
Python	Bahasa utama
Scikit-learn	Algoritma Random Forest
MLflow	Tracking eksperimen
Prometheus	Monitoring metrik model
Grafana	Visualisasi performa model
GitHub Actions	CI/CD pipeline untuk model development

🧪 Hasil Model
Akurasi Model: Dicetak di terminal dan terekam otomatis ke MLflow

Bisa dibandingkan lintas eksperimen lewat UI MLflow Web.

🧭 Roadmap / Pengembangan ke Depan
Mengaktifkan MLflow REST API untuk integrasi deployment

Menerapkan model registry dan model serving

Menambahkan unit tests untuk validasi preprocessing dan evaluasi

Logging ke Prometheus lebih detail (confusion matrix, precision/recall)
```
