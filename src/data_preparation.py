import pandas as pd

from src.utils import cefr_to_numeric, degree_to_numeric


def process_language_match(talent_languages, job_languages):
    """
    Process the language requirements for job and talent.
    Returns a feature score based on the matching algorithm.
    If a must_have language requirement is not met, it is a strong mismatch.

    Args:
    - talent_languages (dict): Dictionary of languages from talent profile.
    - job_languages (dict): Dictionary of languages from job profile.

    Returns:
    - feature_score_languages (float) or -1: Feature score for language matching.
    """
    total_job_languages = len(job_languages)
    points = 0
    for job_language in job_languages:
            job_language_title = job_language['title']
            job_language_level = cefr_to_numeric(job_language['rating'])
            for talent_language in talent_languages:
                talent_language_title = talent_language['title']
                talent_language_level = cefr_to_numeric(talent_language['rating'])
                if talent_language_title == job_language_title:
                    if talent_language_level >= job_language_level:
                        points += 1
                    else:
                        if job_language['must_have']:
                            return -1
    
    feature_score_languages = points/total_job_languages
    return feature_score_languages


def process_job_roles_match(talent_job_roles, job_job_roles):
    """
    Process the job roles requirements for job and talent.
    Returns a feature score based on the matching algorithm.

    Args:
    - talent_job_roles (list): List of job roles from talent profile.
    - job_job_roles (list): List of job roles from job profile.

    Returns:
    - feature_score_job_roles (float): Feature score for job roles matching.
    """
    points = 0
    for job_job_role in job_job_roles:
        if job_job_role in talent_job_roles:
            points += 1
    feature_score_job_roles = points/len(job_job_roles)
    return feature_score_job_roles


def process_seniority_match(talent_seniority, job_seniorities):
    """
    Process the seniority requirements for job and talent.
    Returns a boolean feature score based on the matching algorithm.

    Args:
    - talent_seniority (str): Seniority level in the talent profile.
    - job_seniorities (list): List of seniority levels in the job profile.

    Returns:
    - int: Feature score (boolean) for job seniority matching.
    """
    if talent_seniority in job_seniorities:
        return 1
    else:
        return 0


def process_degree_match(talent_degree, job_min_degree):
    """
    Process the degree requirements for job and talent.
    Returns a feature score based on the matching algorithm.

    Args:
    - talent_degree (str): Degree level in the talent profile.
    - job_min_degree (str): Minimum degree level in the job profile.

    Returns:
    - int: Feature score (boolean) for job seniority matching.
    """
    talent_degree = degree_to_numeric(talent_degree)
    job_degree = degree_to_numeric(job_min_degree)

    if talent_degree >= job_degree:
        return 1
    else:
        return 0


def preprocess_data(data):
    """
    The main function that contains all the pipeline of data processing.

    Args:
    - data (json): This is the input json data.

    Returns:
    - df_data (pandas DataFrame): The processed data in a Pandas DataFrame structure.
    """
    
    df_columns = ['languages',
                  'job_roles',
                  'seniority',
                  'degree',
                  'salary_expectation',
                  'max_salary',
                  'label']

    df_data = pd.DataFrame(columns=df_columns)

    for entry in data:
        feature_language = process_language_match(entry['talent']['languages'], entry['job']['languages'])
        feature_job_roles = process_job_roles_match(entry['talent']['job_roles'], entry['job']['job_roles'])
        feature_seniority = process_seniority_match(entry['talent']['seniority'], entry['job']['seniorities'])
        feature_degree = process_degree_match(entry['talent']['degree'], entry['job']['min_degree'])

        column_values = [feature_language, 
                         feature_job_roles, 
                         feature_seniority, 
                         feature_degree, 
                         entry['talent']['salary_expectation'], 
                         entry['job']['max_salary'], 
                         entry['label']]
    
        df_data.loc[len(df_data)] = column_values
    
    return df_data