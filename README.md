# Fraud Job Posting Detection System

## Overview

This project is a Machine Learning based web application that detects fraudulent job postings using Natural Language Processing (NLP) techniques.

The system analyzes job descriptions and predicts whether a job posting is genuine or fraudulent.

---

## Features

- Detects fraudulent job postings
- NLP-based text preprocessing
- TF-IDF feature extraction
- Machine Learning prediction model
- User-friendly Flask web application

---

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- NLP
- TF-IDF

---

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
└── final_Project.ipynb
```

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/palurusahithi/Fraud-Job-Detection-System.git
```

2. Move into the project folder

```bash
cd Fraud-Job-Detection-System
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
python app.py
```

---

## How It Works

1. User enters job details.
2. Text is cleaned and preprocessed.
3. TF-IDF converts text into numerical features.
4. Machine Learning model predicts:
   - Genuine Job
   - Fraudulent Job

---

## Future Improvements

- Deploy on Render
- Improve model accuracy
- Add user authentication
- Add database integration

---


## Screenshots

### Home Page

![Home Page](screenshots/home.png)

### Prediction Result

![Prediction Result](screenshots/result.png)

## Author

Sahithi Paluru

GitHub:
https://github.com/palurusahithi