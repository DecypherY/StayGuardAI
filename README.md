# StayGuard AI — Hotel Booking Cancellation Predictor

StayGuard AI is a machine learning project that estimates the risk of last-minute hotel booking cancellations using historical booking information.

The project includes data preprocessing, exploratory data analysis, machine learning model development, evaluation, and an interactive Streamlit application.

## Project Objectives

- Analyze historical hotel booking patterns.
- Identify factors associated with last-minute cancellations.
- Build a classification model using Random Forest.
- Evaluate model performance using classification metrics.
- Provide an interactive application for booking risk estimation.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Dataset

The project uses the Hotel Booking Demand dataset, containing 119,390 booking records and 32 original features.

After removing exact duplicate records, 87,396 records remained.

The engineered target variable is `last_minute_cancel`.

Target distribution:

| Class | Records | Percentage |
|---|---:|---:|
| No last-minute cancellation | 80,095 | 91.65% |
| Last-minute cancellation | 7,301 | 8.35% |

The target was constructed using cancellation status and the difference between arrival date and reservation status date.

## Project Workflow

1. Data loading and inspection
2. Duplicate removal and missing-value handling
3. Target variable engineering
4. Exploratory Data Analysis
5. Feature selection
6. Training, validation and test splitting
7. Numerical imputation and categorical encoding
8. Random Forest model training
9. Threshold selection using validation data
10. Final test evaluation
11. Interactive Streamlit application

## Model

**Algorithm:** Random Forest Classifier

- 200 decision trees
- Minimum samples per leaf: 5
- Balanced class weights
- Median imputation for numerical features
- One-Hot Encoding for categorical features

## Model Evaluation

| Metric | Result |
|---|---:|
| ROC-AUC | 0.799 |
| Precision | 28.9% |
| Recall | 41.6% |
| F1-score | 0.341 |
| Average Precision | 0.302 |

These are experimental results from the held-out test set.

## Application

The Streamlit application provides:

- A cancellation-risk dashboard
- An interactive booking risk predictor
- Model architecture and evaluation information

## Limitations

- The dataset contains historical bookings from 2015–2017.
- The target uses `reservation_status_date` as a proxy for cancellation timing.
- The target may include no-show cases.
- Random splitting does not fully simulate future deployment.
- Model scores have not been calibrated as real-world cancellation probabilities.
- The model is an educational demonstration, not a validated operational decision system.

## Running Locally

Clone the repository:

```bash
git clone https://github.com/DecypherY/StayGuardAI.git
cd StayGuardAI
```

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Download the Hotel Booking Demand dataset and place the original CSV at:

`data/hotel_bookings.csv`

Run preprocessing:

```bash
python src/data_preprocessing.py
```

Run model training:

```bash
python src/train.py
```

Launch the application:

```bash
streamlit run app/streamlit_app.py
```

## Author

Developed as a machine learning portfolio project.