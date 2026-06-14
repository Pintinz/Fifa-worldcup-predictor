# FIFA World Cup Predictor ⚽

## Overview

This project is a **machine learning-based football match prediction system** built to predict match outcomes using historical FIFA data and team statistics. It includes a trained ML model and a **Streamlit web application** for interactive predictions.

The model estimates probabilities for:

* Home Win
* Draw
* Away Win
* Over/Under

---

## Features

* Data preprocessing and feature engineering on football datasets
* Machine learning model trained on historical FIFA matches
* Match outcome prediction with probability scores
* Streamlit web interface for easy interaction
* Scalable structure for adding new tournaments and datasets

---

## Project Structure

```
Fifa-worldcup-predictor/
│
├── Fifa.py                  # Streamlit web application
├── Fifa_prediction.ipynb              # Data preprocessing and feature creation
├── scaler.pkl              # Feature scaler
├── requirements.txt        # Project dependencies
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Fifa-worldcup-predictor.git
cd Fifa-worldcup-predictor
```

Create virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## Model Details

* Algorithm: Machine Learning classifier (e.g., Random Forest / XGBoost)
* Input Features:

  * Team form
  * Historical performance
  * Goals scored/conceded
  * Match statistics

Output:

* Probability of Home Win
* Probability of Draw
* Probability of Away Win

---

## Future Improvements

* Add live FIFA dataset API integration
* Improve model accuracy using ensemble learning
* Deploy on Streamlit Cloud or HuggingFace Spaces
* Add player-level statistics

---

## Author

Developed by **Autotechub**

---

## License

This project is open-source and free to use for educational purposes.
