# Governance-Based Security Framework (GBSF) for Biometric Cloud Data

This project is a proof-of-concept implementation of a **Governance-Based Security Framework (GBSF)** designed to secure wearable biometric data (For example: from smartwatches) processed in cloud environments.

The core of this project is a **behavioural biometric anomaly detection system** that uses machine learning to learn a user's unique movement patterns and make real-time, risk-based authentication decisions.

## 1. The Problem
The widespread adoption of wearable devices enables continuous biometric data collection, but this creates significant privacy and security risks when the data is stored and processed in the cloud. A robust framework is required to mitigate these risks and ensure user trust.

## 2. The Solution
This framework proposes a multi-layered security approach. This repository contains the code for the core intelligence module: an **anomaly detection system** built in Python.

This system processes a user's accelerometer and gyroscope data to learn their unique behavioural patterns. A simulated AI agent then uses a "biometric score" from the model to make real-time authentication decisions (ex: *Authenticate*, *Challenge*, or *Deny Access*).

## 3. How It Works
1.  **Data Processing:** A **Python**-based pipeline was engineered to process and aggregate accelerometer and gyroscope features from the dataset.
2.  **Model Training:** A **K-Nearest Neighbours (KNN)** machine learning model was trained on these features to learn and identify unique user patterns.
3.  **Risk-Based Authentication:** The trained model serves as a practical foundation for a continuous authentication system, flagging anomalous behaviour that deviates from the user's learned pattern.

## 4. Tech Stack & Dataset
* **Language:** Python
* **Libraries:** Scikit-learn (for KNN), Pandas, NumPy (for data processing)
* **Dataset:** [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)

## 5. Performance & Results
The prototype successfully demonstrated the feasibility of this approach.
* **Area Under the Curve (AUC):** ~0.7143
* **Equal Error Rate (EER):** ~0.2857

These metrics validate the discriminative power of the behavioural features and confirm the system's viability.

## 6. Compliance & Significance
This work serves as a blueprint for enhancing security and privacy in wearable biometric systems. The framework's design is aligned with key regulatory standards, including:
* **GDPR**
* **ISO/IEC 24745** (Biometric information protection)
