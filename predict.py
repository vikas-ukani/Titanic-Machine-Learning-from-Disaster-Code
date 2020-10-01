# Import necessary packages
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from the current directory
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./train.csv')

# Features extraction for train 
survived = train_data['Survived']
train_data.drop('Survived', axis=1, inplace=True)

# Machine Learning model Implementation
model = DecisionTreeClassifier()

# Model Fitting
model.fit(train_data, survived)
predictions = model.predict(test_data)

# Print the Prediction
print(predictions)
