**✍️ Essay Scoring using Sentence-BERT and Machine Learning**
**Dataset:** [Kaggle – Final Exam Artificial Intelligence 2025 (MATH ITS)](https://www.kaggle.com/competitions/final-exam-artificial-intelligence-2025-math-its/data)  

**🎯 Problem Statement**
Tugas ini bertujuan untuk **memprediksi nilai esai bahasa Inggris secara otomatis** berdasarkan empat aspek penilaian utama:  
- **Task Achievement**  
- **Coherence and Cohesion**  
- **Lexical Resource**  
- **Grammatical Range**  
Tujuan utama proyek ini adalah membangun sistem *automated essay scoring* yang dapat mempercepat proses penilaian dan menjaga konsistensi antar penilai.  

**⚙️ Approach**
🔍 1. Data Preprocessing  
- Menggabungkan kolom `prompt` dan `essay` menjadi satu teks utuh.  
- Menghapus baris dengan nilai target yang kosong.  
- Melakukan *feature extraction* menggunakan dua pendekatan:  
  - **TF-IDF Vectorizer** untuk representasi berbasis kata.  
  - **Sentence-BERT (paraphrase-MiniLM-L6-v2)** untuk representasi berbasis makna (semantic embedding).  
🧠 2. Modeling
Beberapa model regresi diuji untuk memprediksi skor esai secara bersamaan (*multi-output regression*):  
- Random Forest Regressor 
- Multi-Layer Perceptron (MLP)
- XGBoost Regressor
- TF-IDF + Linear Regression (Baseline)
📈 3. Evaluation
Kinerja model diukur menggunakan metrik Mean Squared Error (MSE) untuk menilai seberapa jauh prediksi dari nilai sebenarnya.

**🏆 Results**
**| Model | MSE |**
**|--------|------|**
| Random Forest Regressor | 0.2432 | 
| Multi-Layer Perceptron | 0.3884 | 
| XGBRegressor | 0.4151 |
| F-IDF + Linear Regression | 0.6686 |
> 🔹 Model terbaik adalah Random Forest Regressor dengan nilai MSE terendah, menunjukkan kemampuannya dalam menangkap hubungan kompleks antar fitur teks.

**💡 Insights**
- Sentence-BERT menghasilkan representasi teks yang lebih bermakna dibanding TF-IDF murni.  
- Model berbasis *ensemble* seperti **Random Forest** memberikan hasil paling konsisten.  
- MLP dan XGBoost bekerja baik pada data besar, namun membutuhkan tuning yang lebih kompleks.  

**🧰 Tools & Libraries**
`Python`, `Pandas`, `NumPy`, `Scikit-learn`, `XGBoost`, `Sentence-Transformers`, `Jupyter/Colab`  

🔗 **Run this notebook:** [Google Colab Project Link](https://colab.research.google.com/drive/1Kjbf5ygkp_HoB_UvZbATieBBYUTn_mNJ?usp=sharing)
