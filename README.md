🩺 AI-Powered Medical Diagnosis Assistant with Explainable AI
This project develops a high-performance, explainable, and interactive AI assistant for preliminary medical risk assessment across a broad spectrum of diseases. It utilizes advanced ensemble deep learning and traditional machine learning models, coupled with cutting-edge Explainable AI (XAI) techniques to provide transparent and trustworthy predictions.

✨ Key Features
High-Accuracy Ensemble Model: Developed and optimized a powerful TabNet + XGBoost ensemble model to predict risk across 41 different diseases, achieving a robust 88% accuracy.

Explainable Predictions (XAI): Integrated SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to generate human-readable explanations for every prediction, fostering user trust.

Patient Similarity Recommendations: Provides recommendations based on the similarity of the new patient's symptoms and features to historical patient data.

Interactive Web Application: Deployed a fast, user-friendly, and interactive dashboard using Streamlit that processes real-time risk assessments in less than 2 seconds.

⚙️ Tech Stack
Category	Tools & Libraries
Language	Python
Deep Learning	PyTorch, TabNet
Machine Learning	XGBoost, Scikit-learn
Explainable AI	SHAP, LIME
Deployment	Streamlit
Data/Notebook	Jupyter (.ipynb)

Export to Sheets
🚀 Deployment and Usage
Prerequisites
You need to have Python installed and the following libraries available. Since this project is implemented within a Google Colab notebook (.ipynb), the dependencies are typically installed within the notebook itself.

Clone the Repository

Bash

git clone https://github.com/Hamzah1507/Medical-Diagnosis-AI.git
cd Medical-Diagnosis-AI
Run the Colab Notebook

The entire project, including model training, XAI integration, and the Streamlit setup, is contained within the notebook: Medical_Diagnosis_AI_Assistant.ipynb.

Open the notebook in Google Colab.

Follow the cells sequentially: they handle environment setup, dependency installation (including PyTorch, TabNet, SHAP, etc.), model loading/training, and the final Streamlit deployment.

Launch the Streamlit App (from within Colab)

The last few cells of the Colab notebook handle the deployment using ngrok to tunnel the local Streamlit application to a public URL.

Run the Streamlit cells.

The output will provide a public URL (the ngrok link) to access the interactive web application.

Click the link to start using the AI-Powered Medical Diagnosis Assistant.

💡 Project Structure
The core of the project is the Google Colab notebook, which contains all code and documentation.

Medical-Diagnosis-AI/
├── Medical_Diagnosis_AI_Assistant.ipynb   # Main Google Colab notebook (Training, XAI, and Deployment)
└── README.md                              # Project Readme (You are here)
🤝 Contribution
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions for improving model performance, XAI techniques, or the Streamlit interface.
