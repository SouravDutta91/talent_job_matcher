from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime


class Model:
    def __init__(self, classifier):
        """
        Initialize the Model with a given classifier.

        Parameters:
        - classifier: An instance of a scikit-learn classifier.
        """
        self.classifier = classifier

    def train(self, X_train, y_train):
        """
        Train the classifier on the training data.

        Parameters:
        - X_train: Features of the training data.
        - y_train: Labels of the training data.
        """
        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier on the test data and print evaluation metrics.

        Parameters:
        - X_test: Features of the test data.
        - y_test: Labels of the test data.

        Return:
        - A dictionary of the results.
        """
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Model Evaluation:")
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
        return {'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1}

    def predict(self, X):
        """
        Predict labels for the input features.

        Parameters:
        - X: Features of the data to predict labels for.

        Returns:
        - Predicted labels.
        """
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Predict probability estimates for the input features.

        Parameters:
        - X: Features of the data to predict probabilities for.

        Returns:
        - Probability estimates.
        """
        return self.classifier.predict_proba(X)
    
    def save(self, model_path, name):
        """
        Save the trained model locally.

        Parameters:
        - model_path (str): Filepath to directory where you want to save the model.
        - model_name (str): Name of the model.

        Returns:
        - None
        """
        timestamp = datetime.now()
        joblib.dump(self.classifier, model_path + timestamp.strftime("%Y-%m-%d_%H-%M-%S") + '_' + name + '.pkl')