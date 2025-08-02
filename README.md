#  Wine Quality Prediction Web App

A Flask-based machine learning web application that predicts the **quality of red wine** based on its physicochemical properties. The prediction model is built using **ElasticNet Regression**, which combines the penalties of both Lasso and Ridge regressions.

---

##  Features

- Upload a CSV file with wine properties
- Predict wine quality using a trained ElasticNet Regression model
- Interactive and responsive web UI
- Error handling for invalid input formats
- Fully customizable and easy to deploy

---

##  Machine Learning Model

- **Algorithm:** ElasticNet Regression  
- **Libraries Used:** `scikit-learn`, `pandas`, `numpy`
- **Target Variable:** `quality`
- **Features:**
  - fixed acidity
  - volatile acidity
  - citric acid
  - residual sugar
  - chlorides
  - free sulfur dioxide
  - total sulfur dioxide
  - density
  - pH
  - sulphates
  - alcohol

---

##  Project Structure
```
wine-quality-elasticnet/
│
├── app.py # Flask backend
├── model_train.py # Model training script
├── winequality.csv # Sample dataset
├── elasticnet_model.pkl # Saved ML model
│
├── templates/
│ ├── index.html # Home page
│ └── result.html # Result display page
│
├── static/
│ └── style.css # Custom CSS styles
│
└── README.md # Project documentation
```


---

##  Sample Input Data

Use a CSV file with the following format:

```
fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol
7.4,0.70,0.00,1.9,0.076,11,34,0.9978,3.51,0.56,9.4
```
Note: Do not include the quality column in the uploaded file; it's what we are predicting.

## How to Run Locally
Clone the repository
```
git clone https://github.com/your-username/wine-quality-elasticnet.git
cd wine-quality-elasticnet
```
Install required libraries
```
pip install -r requirements.txt
```
If requirements.txt not provided, install manually:
```
pip install flask pandas scikit-learn
```
Train the model
```
python model_train.py
```
Start the Flask app
```
python app.py
```
Access the app

Go to http://127.0.0.1:5000 in your browser.
## Screenshots
![alt text](<Screenshot 2025-08-02 095536.png>)
![alt text](<Screenshot 2025-08-02 095545.png>)
![alt text](<Screenshot 2025-08-02 095553.png>)