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