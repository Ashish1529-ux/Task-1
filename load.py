# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
# Replace 'tested.csv' with the actual path if it's in a different directory
data = pd.read_csv(r'C:\Users\ADMIN\Desktop\data science\tested.csv')

# Displaying the first few rows of the dataset to understand its structure
print(data.head())

# Step 2: Data Preprocessing

# 2.1: Handle missing values
# Filling missing 'Age' values with the median (for numerical stability)
data['Age'] = data['Age'].fillna(data['Age'].median())

# Filling missing 'Embarked' values with the most frequent value (mode)
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# 2.2: Drop irrelevant columns
# Dropping 'PassengerId', 'Name', 'Ticket', and 'Cabin' as they won't be used in the model
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Step 3: Encode categorical variables
# Converting 'Sex' column into numerical values: male -> 0, female -> 1
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Converting 'Embarked' column into numerical values: C -> 0, Q -> 1, S -> 2
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Step 4: Define Features and Target
# Features (input variables) are all columns except 'Survived'
X = data.drop('Survived', axis=1)

# Target variable is the 'Survived' column
y = data['Survived']

# Step 5: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
# Using RandomForestClassifier as the model
model = RandomForestClassifier(random_state=42)

# Training the model with the training data
model.fit(X_train, y_train)

# Step 7: Make Predictions
# Using the trained model to make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
# Evaluating the model's accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Detailed classification report (precision, recall, f1-score, etc.)
print(classification_report(y_test, y_pred))

# Step 9: Feature Importance (Optional)
# If you want to see which features are most important in predicting survival:
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the feature importance
feature_importances = model.feature_importances_
features = X.columns

# Creating a DataFrame for feature importance
feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sorting the features by importance
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importance
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Feature Importance')
plt.show()

# Step 10: Save the Model (Optional)
# If you want to save the trained model for future use:
import joblib
joblib.dump(model, 'titanic_survival_model.pkl')

# Step 11: Predict on New Data (Optional)
# If you have new data (e.g., test data), you can predict survival like this:
# new_data = pd.read_csv('new_data.csv')  # Load new data
# new_data_predictions = model.predict(new_data)
