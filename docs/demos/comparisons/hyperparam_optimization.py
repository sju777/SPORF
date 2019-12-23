from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def hyperparameter_optimization_random(X, y, *argv):
    """
    Given a classifier and a dictionary of hyperparameters, find optimal hyperparameters using RandomizedSearchCV.

    Parameters
    ----------
    X : numpy.ndarray
        Input data, shape (n_samples, n_features)
    y : numpy.ndarray
        Output data, shape (n_samples, n_outputs)
    *argv : list of tuples (classifier, hyperparameters)
        List of (classifier, hyperparameters) tuples:

        classifier : sklearn-compliant classifier
            For example sklearn.ensemble.RandomForestRegressor, rerf.rerfClassifier, etc
        hyperparameters : dictionary of hyperparameter ranges
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html.

    Returns
    -------
    clf_best_params : dictionary
        Dictionary of best hyperparameters
    """

    clf_best_params = {}

    # Iterate over all (classifier, hyperparameters) pairs
    for clf, params in argv:

        # Run randomized search
        n_iter_search = 10
        random_search = RandomizedSearchCV(
            clf, param_distributions=params, n_iter=n_iter_search, cv=10, iid=False
        )
        random_search.fit(X, y)

        # Save results
        clf_best_params[clf] = random_search.best_params_

    return clf_best_params


def hyperparameter_optimization_grid(X, y, *argv):
    """
    Given a classifier and a dictionary of hyperparameters, find optimal hyperparameters using GridSearchCV.

    Parameters
    ----------
    X : numpy.ndarray
        Input data, shape (n_samples, n_features)
    y : numpy.ndarray
        Output data, shape (n_samples, n_outputs)
    *argv : list of tuples (classifier, hyperparameters)
        List of (classifier, hyperparameters) tuples:

        classifier : sklearn-compliant classifier
            For example sklearn.ensemble.RandomForestRegressor, rerf.rerfClassifier, etc
        hyperparameters : dictionary of hyperparameter ranges
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html. 

    Returns
    -------
    clf_best_params : dictionary
        Dictionary of best hyperparameters
    """

    clf_best_params = {}

    # Iterate over all (classifier, hyperparameters) pairs
    for clf, params in argv:

        # Run grid search
        grid_search = GridSearchCV(
            clf, param_grid=params, cv=10, iid=False
        )
        grid_search.fit(X, y)

        # Save results
        clf_best_params[clf] = grid_search.best_params_

    return clf_best_params
    