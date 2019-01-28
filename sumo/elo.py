from collections import defaultdict
import pandas as pd

def rankings(match_df, rid1_col, rid2_col, result_col):
    ratings_cache = defaultdict(lambda: 1000.0)
    ratings_1 = []
    ratings_2 = []
    indices = []
    for index, row in match_df.iterrows():
        row = row.to_dict()
        rid1 = index[2]
        rid2 = row[rid2_col]

        rid1_rating = ratings_cache[rid1]
        rid2_rating = ratings_cache[rid2]

        indices.append(index)
        ratings_1.append(ratings_cache[rid1])
        ratings_2.append(ratings_cache[rid2])

        result = row[result_col]

        update_weight = update_rating_sigmoid(rid1_rating, rid2_rating, result)
        ratings_cache[rid1] += update_weight
        ratings_cache[rid2] -= update_weight


    ratings_1 = pd.Series(ratings_1, index=pd.MultiIndex.from_tuples(indices))
    ratings_2 = pd.Series(ratings_2, index=pd.MultiIndex.from_tuples(indices))
    c = match_df.copy()
    c['elo'] = ratings_1
    c['elo_opp'] = ratings_2
    return c

def expected_outcome(own_rating, opponent_rating, weight=200.):
    """Compute the expected outcome given two ratings"""
    return 1./(1. + 10.**((opponent_rating - own_rating)/ float(weight)))

def reset_rankings(names):
    BASE_LINE = 1000 # Everybody starts with a 1000 rating
    ratings = pd.Series(np.zeros_like(names.index.values), index=names.index)
    ratings += BASE_LINE
    return ratings

def update_rating_sigmoid(own_rating, opponent_rating, outcome_act, k=20.):
    p = expected_outcome(own_rating, opponent_rating)
    return (outcome_act - p) * k

def update_rating(rating_old, outcome_exp, outcome_act, k=20.):
    return rating_old + k * (np.abs(outcome_act) - np.abs(outcome_exp)) + k * outcome_act

def mean_regression(rating_old, c=0.8):
    return c * rating_old + 1000. * (1 - c)
