import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class GradientBoostingClassifier(object):
    """
    This class implements a Gradient Boosting Classifier using Decision Trees as base learners.

    Parameters:
        n_estimators (int): The number of boosting stages to perform (default: 150)
        learning_rate (float): Controls the contribution of each tree (default: 0.1)
        loss (str): The loss function used for calculating residuals (default: 'categorical_crossentropy')
        max_depth (int): The maximum depth of each decision tree (default: 4)
        max_features (int or None): The number of features to consider when looking for the best split (default: None)
    """

    def __init__(self, n_estimators=40, learning_rate=0.6, loss='log_loss', max_depth=4, max_features=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.max_features = max_features
        self.models_ = []  # Stores the fitted decision tree models
        self.train_preds_ = []  # Stores training predictions for each stage
        self.OoP_preds_ = []  # Stores out-of-pocket (OOP) predictions for each stage

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
        self.X_train_ = X_train
        self.X_test_ = X_test
        self.y_train_ = y_train

        for i in range(self.n_estimators):
            # Create a new decision tree model
            model = DecisionTreeClassifier(random_state=42, max_depth=self.max_depth, max_features=self.max_features)

            # Fit the model on the current training data
            model.fit(self.X_train_, self.y_train_)

            # Store fitted model
            self.models_.append(model)

            # Get predictions from the new tree on the training data
            train_preds = model.predict(self.X_train_)
            self.train_preds_.append(train_preds)

            # Get predictions from the new tree on the testing data
            OoP_preds = model.predict(X_test)
            self.OoP_preds_.append(OoP_preds)

            # Calculate residuals for the next stage (only for boosting stages except the last)
            if i < self.n_estimators - 1:
                y_residual = y_train - train_preds
                self.y_train_ = y_residual

    def predict(self, X):
        preds = np.zeros(X.shape[0])  # Initialize an array to store predictions
        for model, train_preds in zip(self.models_, self.train_preds_):
            # Get predictions from each tree and accumulate them with learning rate
            preds += model.predict(X) * self.learning_rate
        return preds
    
    