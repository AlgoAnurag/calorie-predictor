import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pickle

# Load datasets
calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

# Merge datasets
data = pd.merge(exercise, calories, on='User_ID')

# Encode 'Gender'
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Features and target
X = data.drop(columns=['User_ID', 'Calories'])
y = data['Calories']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBRegressor()
model.fit(X_train, y_train)

# Save the model
with open('calories_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as 'calories_model.pkl'")
