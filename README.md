# California Housing Price Prediction

This project implements a House Price Prediction Model using the California Housing Dataset, inspired by the exercises in the book *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron.

The goal is to build a regression model that predicts the median house value in California districts based on features like median income, housing age, total rooms, and population.

---

## Project Structure

```
house-prediction-project/
│── data/                      # Dataset (small CSV included)
│── notebooks/                 # Jupyter notebooks with EDA & training
│── src/                       # Python scripts for preprocessing & model
│── models/                    # (ignored in Git) trained model pkl files
│── .gitignore
│── README.md
│── requirements.txt
```

---

## Dataset

- Source: California Housing dataset (from Scikit-Learn)  
- Included file: `housing.csv` (small dataset for reproducibility)  

Each row represents a block group in California with the following features:

- `longitude` – Latitude coordinate  
- `latitude` – Longitude coordinate  
- `housing_median_age` – Median age of houses  
- `total_rooms` – Total number of rooms  
- `total_bedrooms` – Total number of bedrooms  
- `population` – Block group population  
- `households` – Total number of households  
- `median_income` – Median income of the block group  
- `median_house_value` – Target variable (house price)

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vishesh-banna0/california-house-prediction.git
   cd california-house-prediction
   ```


2. Run Jupyter Notebook for exploration:
   ```bash
   jupyter notebook notebooks/housing_price.ipynb
   ```

---

## Model Training

The pipeline includes:
- Data preprocessing (handling missing values, feature scaling, encoding categories)  
- Exploratory Data Analysis (EDA)  
- Training models:
  - Linear Regression
  - Decision Trees
  - Random Forest Regressor
- Hyperparameter tuning with GridSearchCV, RandomizedSearchCV 

---

## Trained Model

- The trained model (`my_california_housing_model.pkl`) is not included in this repo due to size limits.  
- You can either train it yourself by running the notebook, or download the pre-trained model from here:  
  [Google Drive Link](https://drive.google.com/file/d/19v_AiRapywL5w2X1sGDK9ivuFajgfybY/view?usp=drive_link) 

---

## Results

- Baseline: Linear Regression – lower accuracy, underfitting  
- Random Forest Regressor – achieved the best performance (lowest RMSE)  
- The model can predict median house values fairly accurately given income and location features  

---

## Future Work

- Add feature engineering for better location-based predictions  
- Try Gradient Boosting or XGBoost for improved performance  
- Deploy as a Flask/FastAPI app for interactive predictions  

---

## Acknowledgements

- Aurélien Géron – *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*  
- Scikit-Learn team for the dataset
