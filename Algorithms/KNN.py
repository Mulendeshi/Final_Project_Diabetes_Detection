import numpy as np

class KNN:
    def __init__(self, k):
        """
        Initialize KNN classifier.

        Parameters:
        - k (int): Number of neighbors to consider for classification.
        """
        self.k = k
        

        

    def fit(self, X_train, y_train, cv = 5):
        """
        Fit the KNN classifier with training data.

        Parameters:
        - X_train (numpy.ndarray): Training features.
        - y_train (numpy.ndarray): Training labels.
        """
        
        self.X_train = X_train
        self.y_train = y_train
        
        

    def predict(self, X_test):
        """
        Predict the labels for test data based on nearest neighbors.
        Parameters:
        - X_test (numpy.ndarray): Test features.
        Returns:
        - predictions (list): Predicted labels for test data.
        """
        predictions = []
        for test_point in X_test:
            distances = np.sqrt(np.sum((self.X_train - test_point)**2, axis=1))  # Calculate Euclidean distance
            nearest_neighbors = np.argsort(distances)[:self.k]  # Get indices of k nearest neighbors
            nearest_labels = self.y_train[nearest_neighbors].astype(int)  # Convert nearest_labels to integers
            predicted_label = np.argmax(np.bincount(nearest_labels))  # Predict the label with the most occurrences
            predictions.append(predicted_label)
        return predictions
    
    
