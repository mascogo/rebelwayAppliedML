import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv')
print(df)
X = df[['x', 'y', 'z']]
y = df['faction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

model = RandomForestClassifier(n_estimators=100, random_state=52)
model.fit(X_train, y_train)

# eval
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
joblib.dump(model, "random_forest_model.pkl")
print("Model saved!")