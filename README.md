# 🩺 AI-Powered Medical Diagnosis Assistant

> A high-performance, explainable, and interactive AI assistant for **preliminary medical risk assessment** across a broad spectrum of diseases — built with ensemble deep learning, advanced XAI techniques, and a real-time Streamlit dashboard.

Built with **TabNet + XGBoost + SHAP/LIME** for transparent, trustworthy, and clinically meaningful predictions.

---

## ✨ Key Features

- 🤖 **Ensemble Deep Learning Model** — TabNet + XGBoost ensemble trained to predict risk across **41 different diseases** with **88% accuracy**
- 🔍 **Explainable AI (XAI)** — Integrated **SHAP** and **LIME** to generate human-readable explanations for every prediction, fostering user trust and clinical transparency
- 👥 **Patient Similarity Recommendations** — Matches new patient profiles to historical data to surface similar cases and contextual risk insights
- ⚡ **Real-Time Risk Assessment** — Interactive Streamlit dashboard processes predictions in **under 2 seconds**
- 📓 **End-to-End Colab Notebook** — Full pipeline from data preprocessing and model training to XAI integration and live deployment, all in one notebook
- 🌐 **Public Deployment via ngrok** — Tunnels the local Streamlit app to a shareable public URL directly from Google Colab

---

## 🛠️ Tech Stack

| Category           | Tools & Libraries                        |
|--------------------|------------------------------------------|
| Language           | Python 3.x                               |
| Deep Learning      | PyTorch, TabNet (pytorch-tabnet)         |
| Machine Learning   | XGBoost, Scikit-learn                    |
| Explainable AI     | SHAP, LIME                               |
| Deployment         | Streamlit + ngrok                        |
| Notebook / Runtime | Google Colab (.ipynb)                    |
| Data Handling      | Pandas, NumPy                            |

---

## 📂 Project Structure

```
Medical-Diagnosis-AI/
├── Medical_Diagnosis_AI_Assistant.ipynb   # Main Colab notebook
│                                          # (Training, XAI & Deployment)
└── README.md                              # Project documentation
```

> The entire pipeline — data processing, model training, XAI integration, and Streamlit deployment — lives inside the single `.ipynb` notebook.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Hamzah1507/Medical-Diagnosis-AI.git
cd Medical-Diagnosis-AI
```

### 2️⃣ Open the Notebook in Google Colab

- Go to [Google Colab](https://colab.research.google.com/)
- Upload or open `Medical_Diagnosis_AI_Assistant.ipynb`
- All dependencies (PyTorch, TabNet, SHAP, LIME, Streamlit, etc.) are installed automatically within the notebook cells

### 3️⃣ Run the Notebook Sequentially

Follow the cells in order — they handle:

- ✅ Environment setup & dependency installation
- ✅ Dataset loading & preprocessing
- ✅ Model training (TabNet + XGBoost ensemble)
- ✅ SHAP & LIME explanation generation
- ✅ Streamlit app setup & ngrok tunnel deployment

### 4️⃣ Launch the Streamlit App

- Run the final deployment cells in the notebook
- An **ngrok public URL** will be printed in the output
- Click the link to open the live **AI Medical Diagnosis Dashboard** in your browser

---

## 🧪 Usage

1. Open the app via the ngrok link generated in Colab
2. **Enter patient symptoms and features** into the interactive form
3. Click **Predict** to receive a real-time disease risk assessment
4. View the **SHAP / LIME explanation** to understand which features drove the prediction
5. Review **similar patient profiles** recommended based on symptom similarity

---

## 🤝 Contributing

Contributions are welcome! If you have ideas to improve model performance, XAI techniques, or the Streamlit interface:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/ImprovementName`)
3. Commit your changes (`git commit -m 'Add ImprovementName'`)
4. Push to the branch (`git push origin feature/ImprovementName`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ using Python, PyTorch & Streamlit</p>
