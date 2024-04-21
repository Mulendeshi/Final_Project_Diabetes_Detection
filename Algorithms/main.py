import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from KNN import KNN
from Gradient_Boost import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# #KNN code beloe

try:
  # Attempt load data assuming there's a header row
  data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
except ValueError: 
  # Load data without assuming no header
  data = np.loadtxt('diabetes.csv', delimiter=',')

# Split data into features and labels
X = data[:, :-1]
y = data[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train KNN classifier
k=3
knn_classifier = KNN(k)
knn_classifier.fit(X_train, y_train, cv=5)

# Predict using KNN classifier
knn_predictions = knn_classifier.predict(X_test)

# Calculate accuracy of KNN predictions
knn_accuracy = accuracy_score(y_test, knn_predictions)
print('KNN Accuracy with K= {k} :', knn_accuracy)

# Calculate confusion matrix for KNN
knn_cm = confusion_matrix(y_test, knn_predictions)

# Plot confusion matrix for KNN
plt.figure(figsize=(8, 6))
sns.heatmap(knn_cm, annot=True, cmap='Blues', fmt='g')
plt.title(f'Confusion Matrix for KNN Classifier with K={k}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# THE Gradient Boost code


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Diabetes dataset
diabetes = pd.read_csv('diabetes.csv', skiprows=1)
X = diabetes.values[:, :-1]
y = diabetes.iloc[:, -1]

# Convert continuous target variable to binary classes
y_binary = np.where(y > np.mean(y), 1, 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=60)

# Initialize the GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=40, learning_rate=0.5, loss='log_loss')

# Fit the model to the training data
gbc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = gbc.predict(X_test)

# Convert predicted probabilities to binary classes
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Print the accuracy score
print('Gradient Accuracy:', np.mean(y_pred_binary == y_test))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Gradient Confusion Matrix')
plt.show()