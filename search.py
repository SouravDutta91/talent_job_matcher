import joblib

from src.data_preparation import preprocess_data


class Search:
    def __init__(self, model_path):
        """
        __init__ constructor to load the trained model.

        Args:
        - model_path (str): This is the filepath to the trained model.
        """
        self.model = joblib.load(model_path)


    def match(self, talent: dict, job: dict) -> dict:
        """
        This method takes a talent and job as input and uses the machine learning
        model to predict the label. Together with a calculated score, the dictionary
        returned has the following schema:

        {
          "talent": ...,
          "job": ...,
          "label": ...,
          "score": ...
        }

        Args:
        - talent (dict): Dictionary with a single talent profile.
        - job (dict): Dictionary with a single job profile.

        Returns:
        - result (dict): Dictionary with the result and score for the input talent-job pair.
        """
        
        data = [{'talent': talent,
                 'job': job,
                 'label': 0}]
        
        df_data = preprocess_data(data)

        X_test = df_data.drop(columns=['label'])

        y_pred = self.model.predict(X_test)[0]
        score = self.model.predict_proba(X_test)[0][1]
        
        # Construct the result dictionary
        result = {
            "talent": talent,
            "job": job,
            "label": bool(y_pred),
            "score": score
        }
        return result
    

    def match_bulk(self, talents: list[dict], jobs: list[dict]) -> list[dict]:
        """
        This method takes a multiple talents and jobs as input and uses the machine
        learning model to predict the label for each combination. Together with a
        calculated score, the list returned (sorted descending by score!) has the
        following schema:
        
        [
          {
            "talent": ...,
            "job": ...,
            "label": ...,
            "score": ...
          },
          {
            "talent": ...,
            "job": ...,
            "label": ...,
            "score": ...
          },
          ...
        ]

        Args:
        - talents (list[dict]): List of dictionaries of talent profiles.
        - jobs (list[dict]): List of dictionaries of job profiles.

        Returns:
        - results (list[dict]): List of dictionaries with results and scores for each talent-job combination.
        """
        
        results = []
        for talent in talents:
            for job in jobs:
                result = self.match(talent, job)
                results.append(result)

        # Sort results by score in descending order
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results