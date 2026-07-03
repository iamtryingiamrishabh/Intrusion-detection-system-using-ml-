# Intrusion Detection System using Machine Learning

A machine learning-based approach to detect malicious network activity and potential intrusions.

## 🚀 Features
* Preprocesses and cleans network traffic datasets.
* Trains Machine Learning models to classify traffic as benign or malicious.
* Includes pre-trained model pipelines (`.pkl` files) for quick deployment.

## 🛠️ Project Structure
* `main.py`: The core script for running the application.
* `sample.py`: Script for testing samples against the models.
* `ids_model_enhanced.pkl`: Saved trained machine learning model.

## 📋 Prerequisites & Dataset
Because the datasets are too large for GitHub, you will need to download them separately:
* **CICIDS Dataset**: `cicids.csv`
* **NSL-KDD Dataset**: `KDDTest+.txt` / `KDDTest+.arff`

## 💻 How to Run
1. Install required libraries (like pandas, scikit-learn, etc.).
2. Run the main script:
   ```bash
   python main.py