from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "energy_kcal",
    "protein_g",
    "fat_g",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "sodium_mg",
]


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier implemented from scratch with numpy.

    Assumes each feature follows a Gaussian (normal) distribution within
    each class.  At prediction time the log-posterior is computed as:

        log P(c | x) ∝ log P(c) + Σ_j log N(x_j ; μ_{cj}, σ²_{cj})

    where N(·) is the Gaussian PDF evaluated in log-space for numerical
    stability.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        """
        Parameters
        ----------
        var_smoothing:
            Small constant added to per-class variances to avoid division by
            zero when a feature has zero variance within a class.
        """
        self.var_smoothing = var_smoothing
        self._classes: np.ndarray | None = None
        self._log_priors: np.ndarray | None = None
        self._means: np.ndarray | None = None  # shape (n_classes, n_features)
        self._vars: np.ndarray | None = None   # shape (n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianNaiveBayes:
        """
        Fit the classifier.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : integer class labels of shape (n_samples,)
        """
        self._classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self._classes)

        self._log_priors = np.zeros(n_classes)
        self._means = np.zeros((n_classes, n_features))
        self._vars = np.zeros((n_classes, n_features))

        for i, c in enumerate(self._classes):
            X_c = X[y == c]
            self._log_priors[i] = np.log(len(X_c) / len(X))
            self._means[i] = X_c.mean(axis=0)
            self._vars[i] = X_c.var(axis=0) + self.var_smoothing

        return self

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log P(X | class) for every class.

        Returns array of shape (n_samples, n_classes).
        """
        log_likelihoods = np.zeros((X.shape[0], len(self._classes)))
        for i in range(len(self._classes)):
            mu = self._means[i]
            var = self._vars[i]
            # Sum of log Gaussian PDF across features
            log_likelihoods[:, i] = (
                -0.5 * np.sum(np.log(2.0 * np.pi * var))
                - 0.5 * np.sum((X - mu) ** 2 / var, axis=1)
            )
        return log_likelihoods

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for each sample.

        Returns array of shape (n_samples, n_classes).
        """
        log_posterior = self._log_likelihood(X) + self._log_priors
        # Subtract row-wise max for numerical stability before exp
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        probs = np.exp(log_posterior)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most probable class label for each sample."""
        return self._classes[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Helpers used by main.py
# ---------------------------------------------------------------------------

def train_preference_classifier(
    foods_df: pd.DataFrame,
    preferences: dict[str, set[str]],
) -> GaussianNaiveBayes:
    """
    Train a GaussianNaiveBayes preference classifier on the food dataset.

    Foods whose category matches a liked category are labelled 1 (positive);
    foods matching a disliked category are labelled 0 (negative).  Nutritional
    features are used so the model can generalise to foods in unknown
    categories.

    Parameters
    ----------
    foods_df    : the full processed foods DataFrame
    preferences : dict with keys "liked_categories" and "disliked_categories"
                  (sets of strings)

    Returns
    -------
    A fitted GaussianNaiveBayes instance.
    """
    liked = {c.lower() for c in preferences.get("liked_categories", set())}
    disliked = {c.lower() for c in preferences.get("disliked_categories", set())}

    cat_lower = foods_df["category"].str.lower()
    mask_liked = cat_lower.apply(lambda c: any(k in c for k in liked))
    mask_disliked = cat_lower.apply(lambda c: any(k in c for k in disliked))

    labeled_df = pd.concat([
        foods_df[mask_liked].assign(_label=1),
        foods_df[mask_disliked].assign(_label=0),
    ])

    X = labeled_df[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
    y = labeled_df["_label"].to_numpy(dtype=int)

    clf = GaussianNaiveBayes()
    clf.fit(X, y)
    return clf


def predict_preference_scores(
    foods_df: pd.DataFrame,
    clf: GaussianNaiveBayes,
) -> np.ndarray:
    """
    Return P(liked=1) for every food in foods_df.

    Parameters
    ----------
    foods_df : the full processed foods DataFrame
    clf      : a fitted GaussianNaiveBayes from train_preference_classifier

    Returns
    -------
    1-D numpy array of probabilities, one per row in foods_df.
    """
    X = foods_df[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
    liked_idx = int(np.where(clf._classes == 1)[0][0])
    return clf.predict_proba(X)[:, liked_idx]
