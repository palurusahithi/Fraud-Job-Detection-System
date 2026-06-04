# Fraud Job Posting Detection System

## Overview

This project is a Machine Learning-based web application that detects fraudulent job postings using Natural Language Processing (NLP) techniques.

The system analyzes job descriptions and predicts whether a job posting is genuine or fraudulent.

## Features

- Detects fraudulent job postings
- NLP-based text preprocessing
- TF-IDF feature extraction
- BERT embeddings
- Machine Learning prediction model
- OCR support for image-based job postings
- User-friendly Flask web application

## Technologies Used

- Python
- Flask
- Scikit-learn
- PyTorch
- Transformers (BERT)
- OpenCV
- Tesseract OCR
- Pandas
- NumPy
- NLP
- TF-IDF

## Project Structure

```text
Fraud-Job-Detection-System/
│
├── app.py
├── requirements.txt
├── mlp_model.joblib
├── scaler.joblib
├── tfidf.joblib
├── threshold.joblib
├── templates/
├── screenshots/
└── final_Project.ipynb
```

## Installation

### Clone the repository

```bash
git clone https://github.com/palurusahithi/Fraud-Job-Detection-System.git
```

### Move into the project folder

```bash
cd Fraud-Job-Detection-System
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

## How It Works

1. User enters job details or uploads an image.
2. OCR extracts text from uploaded images.
3. Text is cleaned and preprocessed.
4. TF-IDF and BERT generate features.
5. Machine Learning model predicts:
   - Genuine Job
   - Fraudulent Job

## Screenshots

### Home Page

![Home Page](screenshots/home.png)

### Prediction Result

![Prediction Result](screenshots/result.png)

## Future Improvements

- Deploy on Render
- Improve model accuracy
- Add user authentication
- Add database integration

## Author

**Sahithi Paluru**

GitHub: https://github.com/palurusahithi