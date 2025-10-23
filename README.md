
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
- Random Forest Regressor : 0.2432 
- Multi-Layer Perceptron : 0.3884
- XGBRegressor : 0.4151 
- F-IDF + Linear Regression : 0.6686 
> 🔹 Model terbaik adalah Random Forest Regressor dengan nilai MSE terendah, menunjukkan kemampuannya dalam menangkap hubungan kompleks antar fitur teks.

**💡 Insights**
- Sentence-BERT menghasilkan representasi teks yang lebih bermakna dibanding TF-IDF murni.  
- Model berbasis *ensemble* seperti **Random Forest** memberikan hasil paling konsisten.  
- MLP dan XGBoost bekerja baik pada data besar, namun membutuhkan tuning yang lebih kompleks.  

**🧰 Tools & Libraries**
`Python`, `Pandas`, `NumPy`, `Scikit-learn`, `XGBoost`, `Sentence-Transformers`, `Jupyter/Colab`  

🔗 **Run this notebook:** [Google Colab Project Link](https://colab.research.google.com/drive/1Kjbf5ygkp_HoB_UvZbATieBBYUTn_mNJ?usp=sharing)



# 🩺 Essay Scoring using Sentence-BERT and Machine Learning
This project aims to classify diabetes into multiple categories using a dataset from [Kaggle – Final Exam Artificial Intelligence 2025 (MATH ITS)](https://www.kaggle.com/competitions/final-exam-artificial-intelligence-2025-math-its/data).  
Machine learning algorithms are applied to analyze patient data and predict the level or type of diabetes based on several medical attributes.

---

## Open in Google Colab
Click the badge below to run the notebook directly in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malonasntr/machine-learning-diabetes/blob/015475f970fb07bc2b9f40c2bf5d4402c9c2295b/machine_learning.ipynb)

---

## Dataset
The dataset includes multiple attributes such as:
| Feature | Description |
|--------|-----------|
| Gender | The biological sex of the individual. Usually encoded as: 0 = Female, 1 = Male. Gender can influence diabetes risk due to hormonal and lifestyle differences |
| AGE| The age of the individual in years. Age is a critical risk factor for diabetes, as the likelihood increases with age, especially after 45 |
| Urea | Measurement of urea in the blood (mg/dL). High levels may indicate kidney dysfunction, a common complication of diabetes. Normal range: ~7–20 mg/dL |
| Cr (Creatinine) | Measures the level of creatinine in the blood (mg/dL). Creatinine is an indicator of kidney function; elevated levels may suggest impaired kidney function often associated with diabetes. Normal range: 0.6–1.3 mg/dL |
| HbA1c (Glycated Hemoglobin) | A key indicator of average blood glucose levels over the past 2–3 months, expressed as a percentage. (Normal: <5.7% ; Prediabetic: 5.7–6.4% ; Diabetic: ≥6.5%) |
| Chol (Cholesterol) | Total cholesterol in the blood (mg/dL). High cholesterol increases the risk of cardiovascular disease and is commonly seen in diabetic individuals. Normal range: <200 mg/dL |
| TG (Triglycerides) | Measures the amount of fat in the blood (mg/dL). High triglycerides are associated with insulin resistance and metabolic syndrome. Normal range: <150 mg/dL |
| HDL (High-Density Lipoprotein) | The “good” cholesterol (mg/dL). Higher levels are beneficial, helping remove excess cholesterol from the bloodstream. (Ideal: >40 mg/dL (men) ; >50 mg/dL (women)) |
| LDL (Low-Density Lipoprotein) | The “bad” cholesterol (mg/dL). High levels contribute to plaque buildup in arteries. Optimal range: <100 mg/dL |
| VLDL (Very Low-Density Lipoprotein) | Another type of “bad” cholesterol that carries triglycerides. Often estimated as TG/5. Normal range: 2–30 mg/dL |
| BMI (Body Mass Index) | A measure of body fat based on height and weight (kg/m²). Obesity (BMI ≥30) is a major risk factor for Type 2 diabetes (- Underweight: <18.5 ; Normal: 18.5–24.9 ; Overweight: 25–29.9 ; Obese: ≥30) |
| Class | Target label representing diabetes status. Encoded as: 0 = Non-Diabetic, 1 = Diabetic, 2 = Prediabetic. This is the variable the model aims to predict |

---

## Machine Learning Workflow
1. **Data Preprocessing**
   - Check duplicate and missing values 
   - Normalize/scale numerical features  

2. **Model Training**
   - Decision Tree  
   - Random Forest  
   - Support Vector Machine (SVM)  

3. **Model Evaluation**
   - Confusion Matrix  
   - Accuracy, Precision, Recall, F1-Score  

---

## Results Summary

| Model | Accuracy | Notes |
|--------|-----------|--------|
| Decision Tree | 0.9811 | Produced high accuracy with minor misclassification in the “Prediabetic” class; performs well but shows potential signs of overfitting |
| Random Forest | 1.0000 | Successfully predicted all classes without any errors, demonstrating excellent generalization capability |
| SVM | 0.9057 | Performed reasonably well but struggled to distinguish the “Prediabetic” class; sensitive to feature scaling and kernel parameter settings |

---

