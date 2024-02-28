import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold


def load_json_data(data_file_path):
    """
    Load the json data.

    Parameters:
    - data_file_path (str): Path to the local json data file.

    Returns:
    - data (json): JSON object of the file.
    """
    with open(data_file_path, 'r') as f:
        data = json.load(f)
    return data


def cefr_to_numeric(level):
    """
    Convert CEFR language levels to numerical levels.

    Parameters:
    - level (str): CEFR language level (e.g., 'A1', 'A2', 'B1', 'B2', 'C1', 'C2').

    Returns:
    - int: Numerical level corresponding to the CEFR language level.
    """
    if level == 'A1':
        return 1
    elif level == 'A2':
        return 2
    elif level == 'B1':
        return 3
    elif level == 'B2':
        return 4
    elif level == 'C1':
        return 5
    elif level == 'C2':
        return 6
    else:
        return None


def degree_to_numeric(degree):
    """
    Convert degree levels to numerical levels.

    Parameters:
    - level (str): Degree level (e.g., 'none', 'apprenticeship', 'bachelor', 'master', 'doctorate').

    Returns:
    - int: Numerical level corresponding to the degree level.
    """
    if degree == 'none':
        return 1
    elif degree == 'apprenticeship':
        return 2
    elif degree == 'bachelor':
        return 3
    elif degree == 'master':
        return 4
    elif degree == 'doctorate':
        return 5


def train_test_split_data(df_data, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing data sets in 80:20 ratio by default.

    Args:
        df_data (Pandas DataFrame): Processed data to be used for training and testing the ML model(s).

    Returns:
    - Training and testing data sets.
    """
    X = df_data.drop(columns=['label'])
    y = df_data['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def perform_kfold_cross_validation(X_train, y_train, models, n_splits=10):
    """
    Performs k-fold cross-validation.

    Args:
        X_train (_type_): Training data features.
        y_train (_type_): Training data labels.
        models (dict): Dictionary wiht list of models.

    Returns:
        results (dict): Dictionary with results of Cross-Validation.
    """
    results = {}
    for name, model in models.items():
        kfold = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results[name] = cv_results
        print(f'{name}: Mean accuracy: {cv_results.mean()}, Std deviation: {cv_results.std()}')
    return results


def plot_cross_val_results(cross_val_results):
    """
    Plot box plots of cross-validation results for each model.

    Parameters:
    - cross_val_results (dict): Dictionary containing cross-validation results for each model.
                                Keys are model names, values are arrays of cross-validation scores.

    Returns:
    - None
    """
    plt.figure(figsize=(25, 8))
    plt.boxplot(cross_val_results.values(), labels=cross_val_results.keys())
    plt.title('Cross-validation Results')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    # plt.savefig('images\cv.png', dpi=300)
