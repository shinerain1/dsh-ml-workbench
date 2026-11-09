"""Example sklearn factory for agent-authored estimator changes."""

from sklearn.linear_model import LogisticRegression, Ridge


def make_estimator(task, params, seed):
    if task == "classification":
        return LogisticRegression(random_state=seed, max_iter=1000, **params)
    return Ridge(random_state=seed, **params)
