import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = pd.read_csv('Wine_quality.csv')

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ElasticNet(alpha=0.1, l1_ratio=0.5))  
])


pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'elasticnet_model.pkl')
print("✅ Model trained and saved as elasticnet_model.pkl")
