import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


class StackingClassifier:
    def __init__(self,base_learners, meta_learner):
        self.base_learners = base_learners
        self.meta_learner = meta_learner

    def fit_base_learners(self, X_train, y_train):
        """
        Fit each base learner on the training data.

        Args:
            X_train (np.ndarray): The training features.
            y_train (np.ndarray): The training labels.
        """
        for learner in self.base_learners:
            learner.fit(X_train, y_train)  # Include y_train argument


    def predict_base_learners(self, X_test):
        return np.column_stack([learner.predict(X_test) for learner in self.base_learners])

    def fit_meta_learner(self, X_train, y_train):
        # Predict using base learners on the training data
        self.meta_learner.fit(X_train, y_train)
    
        # Fit the meta learner using the base learner predictions and the corresponding training labels
        self.meta_learner.fit(base_predictions, y_train)
    def predict(self, X_test):
        base_predictions = self.predict_base_learners(X_test)
        return self.meta_learner.predict(base_predictions)
    

from Gradient_Boost import GradientBoostingClassifier
from KNN import KNN

# Intialize base learner
gbc = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_depth=3)
knn = KNN(k=13)

# to initialize meta Learner
# meta_learner = LogisticRegression(C=0.1, solver='liblinear')
meta_learner = RandomForestClassifier(n_estimators=42, max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1)

#initalise stacking classfier
stacking_model= StackingClassifier(base_learners=[gbc, knn], meta_learner= meta_learner)

# Initialize imputer for handling missing values
# imputer = SimpleImputer(strategy='most_frequent')
# Load data 
try:
  # Attempt load data assuming there's a header row
  data = np.loadtxt('newdiabetes.csv', delimiter=',', skiprows=1)
except ValueError: 
  # Load data without assuming no header
  data = np.loadtxt('newdiabetes.csv', delimiter=',')

# Split data into features and labels
X = data[:, :-1]
y = data[:, -1]

#features to impute
# features_to_impute = np.arange(1, X.shape[1])  # all columns except 0 and last
# X_imputed = np.copy(X)
# X_imputed[:, features_to_impute] = imputer.fit_transform(X[:, features_to_impute])



# Save imputed data to CSV file (optional)
# np.savetxt('dai.csv', np.around(X_imputed), delimiter=',', fmt='%f')
# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=60)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Fit base learners
stacking_model.fit_base_learners(X_train, y_train)

# Train the meta-learner

base_predictions = stacking_model.predict_base_learners(X_train)
stacking_model.fit_meta_learner(base_predictions, y_train)

# Make predictions on the test data
y_pred = stacking_model.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacked Ensemble Accuracy: {accuracy:.4f}")


# Convert predicted probabilities to binary classes
threshold = 0.5
y_pred_binary = np.where(y_pred > threshold, 1, 0)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Plot confusion matrix
plt.figure(figsize=(4, 2))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Stacked Confusion Matrix')
plt.show()
print(conf_matrix)



# Prompt user to input values for the diabetes features
print("Enter the following information:")
pregnancies = float(input("Number of Pregnancies: "))
glucose = float(input("Glucose Level: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin Level: "))
bmi = float(input("BMI: "))
diabetes_pedigree_function = float(input("Diabetes Pedigree Function: "))
Age = int(input("Age: "))
# Prepare input data
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, Age]])

# Make predictions using the stacking model
prediction = stacking_model.predict(input_data)

# Display the prediction result
if prediction == 0:
    print("Based on the provided information, the Patient is not Diabetic")
else:
    print("Based on the provided information, the Patient is Diabetic")










