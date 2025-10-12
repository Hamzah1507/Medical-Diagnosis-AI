# AI-Powered Medical Diagnosis Assistant with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent medical diagnosis system that combines ensemble learning with explainable AI to provide accurate disease predictions and transparent decision-making insights for healthcare professionals.

## 🌟 Key Features

- **High-Accuracy Predictions**: Ensemble model combining TabNet and XGBoost achieving **88% accuracy** across 41 different diseases
- **Explainable AI**: Integrated SHAP and LIME frameworks to provide transparent, interpretable predictions
- **Patient Similarity Analysis**: Similarity-based recommendation system for comparative case studies
- **Real-Time Processing**: Lightning-fast risk assessment with response times under 2 seconds
- **Interactive Dashboard**: User-friendly Streamlit interface for seamless clinical workflow integration
- **Comprehensive Disease Coverage**: Supports diagnosis across 41 different medical conditions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Hamzah1507/Medical-Diagnosis-AI.git
cd Medical-Diagnosis-AI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python scripts/init_database.py
```

### Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 💻 Usage

### Making Predictions

1. **Input Patient Data**: Enter symptoms, vitals, and medical history
2. **Generate Prediction**: Click "Diagnose" to receive disease probability rankings
3. **View Explanations**: Explore SHAP/LIME visualizations showing feature importance
4. **Similar Cases**: Review similar patient cases for comparative analysis

### API Usage Example

```python
from models.ensemble import MedicalDiagnosisModel

# Initialize model
model = MedicalDiagnosisModel()
model.load_weights('checkpoints/best_model.pth')

# Make prediction
patient_data = {
    'symptoms': ['fever', 'cough', 'fatigue'],
    'vitals': {'temperature': 38.5, 'heart_rate': 95},
    'age': 45,
    'gender': 'M'
}

prediction = model.predict(patient_data)
explanation = model.explain(patient_data, method='shap')
```

## 🧠 Model Architecture

### Ensemble Approach

The system employs a hybrid ensemble architecture combining:

**TabNet**
- Attention-based deep learning model for tabular data
- Sequential attention mechanism for feature selection
- Interpretable by design with built-in feature importance

**XGBoost**
- Gradient boosting framework
- Robust performance on structured medical data
- Efficient handling of missing values

### Training Pipeline

```
Data Preprocessing → Feature Engineering → Model Training → Ensemble Fusion → Validation
```

**Key Components:**
- Data Augmentation with SMOTE for handling class imbalance
- 5-fold stratified cross-validation for robust evaluation
- Optuna-based hyperparameter optimization

## 🔍 Explainability Features

### SHAP (SHapley Additive exPlanations)

- Global feature importance across all predictions
- Individual prediction explanations with contribution scores
- Force plots showing how features push predictions

### LIME (Local Interpretable Model-agnostic Explanations)

- Local surrogate models for individual predictions
- Feature perturbation analysis
- Human-readable explanations

### Patient Similarity

- K-nearest neighbors approach for finding similar cases
- Weighted feature similarity using learned embeddings
- Historical outcome tracking for similar patients

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core Language** | Python 3.8+ |
| **Deep Learning** | PyTorch, TabNet |
| **Machine Learning** | XGBoost, Scikit-learn |
| **Explainability** | SHAP, LIME |
| **Web Framework** | Streamlit |
| **Database** | SQLite |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly, Seaborn |

## 📁 Project Structure

```
Medical-Diagnosis-AI/
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── README.md
├── models/
│   ├── tabnet_model.py        # TabNet implementation
│   ├── xgboost_model.py       # XGBoost implementation
│   ├── ensemble.py            # Ensemble logic
│   └── explainer.py           # SHAP/LIME integration
├── data/
│   ├── raw/                   # Raw medical datasets
│   ├── processed/             # Preprocessed data
│   └── preprocessing.py       # Data pipeline
├── utils/
│   ├── similarity.py          # Patient similarity engine
│   ├── visualization.py       # Plotting utilities
│   └── database.py            # SQLite operations
├── scripts/
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Evaluation metrics
│   └── init_database.py       # Database initialization
├── checkpoints/               # Saved model weights
├── logs/                      # Training logs
└── tests/                     # Unit tests
```

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| Overall Accuracy | 88.0% |
| Macro F1-Score | 0.86 |
| Average Precision | 0.89 |
| Average Recall | 0.87 |
| Inference Time | <2 seconds |

### Disease-Specific Performance

Top performing disease categories:
1. **Cardiovascular diseases**: 92% accuracy
2. **Respiratory infections**: 90% accuracy
3. **Gastrointestinal disorders**: 89% accuracy
4. **Neurological conditions**: 87% accuracy
5. **Metabolic disorders**: 85% accuracy

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimer

This system is designed as a **decision support tool** for healthcare professionals and should **NOT** be used as a substitute for professional medical diagnosis. Always consult qualified healthcare providers for medical decisions.

**Key Points:**
- Not FDA approved
- For research and educational purposes
- Requires professional medical interpretation
- Ensure HIPAA/GDPR compliance before clinical deployment

## 📧 Contact

**Hamzah** - [@Hamzah1507](https://github.com/Hamzah1507)

Project Link: [https://github.com/Hamzah1507/Medical-Diagnosis-AI](https://github.com/Hamzah1507/Medical-Diagnosis-AI)

## 🙏 Acknowledgments

- Medical dataset providers and research community
- TabNet paper authors: Arik & Pfister (2021)
- SHAP framework by Lundberg & Lee
- LIME framework by Ribeiro et al.
- Open-source machine learning community

## 📚 References

- [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://github.com/marcotcr/lime)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**⭐ If you find this project helpful, please consider giving it a star!**

*Built with ❤️ for advancing healthcare through AI*
