import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from stack import StackingClassifier
from Gradient_Boost import GradientBoostingClassifier
from KNN import KNN

# Intialize base learner
gbc = GradientBoostingClassifier(n_estimators=40, learning_rate=0.5, loss='log_loss')
knn = KNN(k=13)

# to initialize meta Learner
meta_learner = GridSearchCV(RandomForestClassifier(), cv=5, param_grid={
    'n_estimators': [50, 100, 150, 200],  # Balanced grid with more values
    'max_depth': [3, 4, 5, 6],               # Balanced grid with more values
    
}, scoring='accuracy', verbose=1)
#initalise stacking classfier
stacking_model= StackingClassifier(base_learners=[gbc, knn], meta_learner= meta_learner)

# Initialize imputer for handling missing values
imputer = SimpleImputer(strategy='most_frequent')
# Load data 
try:
  # Attempt load data assuming there's a header row
  data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
except ValueError: 
  # Load data without assuming no header
  data = np.loadtxt('diabetes.csv', delimiter=',')

# Split data into features and labels
X = data[:, :-1]
y = data[:, -1]

#features to impute
features_to_impute = np.arange(1, X.shape[1])  # all columns except 0 and last
X_imputed = np.copy(X)
X_imputed[:, features_to_impute] = imputer.fit_transform(X[:, features_to_impute])

# Save imputed data to CSV file (optional)

# Save imputed data to CSV file (optional)
np.savetxt('dai.csv', np.around(X_imputed, decimals=1), delimiter=',', fmt='%f')
# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=60)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)



# Fit base learners
stacking_model.fit_base_learners(X_train, y_train)

# Train the meta-learner

base_predictions = stacking_model.predict_base_learners(X_train)
stacking_model.fit_meta_learner(base_predictions, y_train)





# Make predictions on the test data
y_pred = stacking_model.predict(X_test)

# Evaluate the performance (e.g., accuracy_score)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacked Ensemble Accuracy: {accuracy:.4f}")


# Convert predicted probabilities to binary classes
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Plot confusion matrix
plt.figure(figsize=(4, 2))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Stacked Confusion Matrix')
plt.show()
# print(conf_matrix)


