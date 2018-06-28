import numpy as np

import sys
sys.path.insert(0, '..')
from utils import one_hot

type_weights = {
    "visited": 0.6,
    "typed": 0.2,
    "bookmarked": 0.2,
    #"other_type": 0.1
}

recency_weights = {
    "4-days": 0.03,
    "14-days": 0.05,
    "31-days": 0.1,
    "90-days": 0.32,
    "other_recency": 0.5
}

recency_weights = {
    "4-days": 0.15,
    "14-days": 0.15,
    "31-days": 0.15,
    "90-days": 0.2,
    "other_recency": 0.35
}

def combine_dicts_multiplicatively(dict1, dict2):
    """
    Returns a new dict where the keys consist of all pairs of keys from the input
    dictionaries and the values correspond to the respective multiplied values.
    """
    weights = {}

    for key1, weight1 in dict1.items():
        for key2, weight2 in dict2.items():
            key = (key1, key2)
            weight = weight1 * weight2
            weights[key] = weight
            
    return weights

weights = combine_dicts_multiplicatively(type_weights, recency_weights)

def sample_weighted(num_samples, weight_dict):
    """Randomly sample from a dict using the values as probabilities"""
    num_choices = len(weight_dict)
    choice_weights = weight_dict.values()
    samples = np.random.choice(num_choices, num_samples, p=choice_weights)
    return one_hot(num_choices, samples)

def sample_url_features(num_samples):
    return sample_weighted(num_samples, weights)

type_points = {
    "visited": 1.2,
    "typed": 2,
    "bookmarked": 1.4,
    #"other_type": 0
}

recency_points = {
    "4-days": 100,
    "14-days": 70,
    "31-days": 50,
    "90-days": 30,
    "other_recency": 10
}

frecency_points_dict = combine_dicts_multiplicatively(type_points, recency_points)
key_order = weights.keys()
frecency_points = np.array([frecency_points_dict[key] for key in key_order])

def frecency(url_features):
    return url_features.dot(frecency_points)

def sample(num_samples):
    X = sample_url_features(num_samples)
    y = frecency(X)
    return X, y

def sample_num_options(n):
    num_options = np.random.normal(loc=10, scale=4, size=(n))
    num_options = np.maximum(num_options, 1)
    return np.int32(num_options)

def sample_suggestions_normal(n):
    num_options = sample_num_options(n)
    data = map(sample, num_options)
    X, y = zip(*data)
    return X, y

def sample_suggestions_spark(n):
    num_options = sample_num_options(n)    
    data = sc.parallelize(num_options).map(sample).collect()
    X, y = zip(*data)
    return X, y

def sample_suggestions(n):
    if n > 1000:
        return sample_suggestions_spark(n)
    else:
        return sample_suggestions_normal(n)