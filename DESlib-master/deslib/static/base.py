# coding=utf-8

from abc import abstractmethod, ABCMeta

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble, BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_random_state


class BaseStaticEnsemble(BaseEstimator, ClassifierMixin):
    """Base class for static ensembles.

    All static ensemble techniques should inherit from this class.

    Warning: This class should not be instantiated directly, use derived
    classes instead.

    Parameters
    ----------
    pool_classifiers : list of classifiers (Default = None)
        The generated_pool of classifiers trained for the corresponding
        classification problem. Each base classifiers should support the method
        "predict". If None, then the pool of classifiers is a bagging
        classifier.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : int, default=-1
        The number of parallel jobs to run. None means 1 unless in
        a joblib.parallel_backend context. -1 means using all processors.
        Doesn’t affect fit method.

    References
    ----------
    Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
    John Wiley & Sons, 2004.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, mode='selection', voting='hard',
                 needs_proba=False, random_state=None,
                 knn_classifier='knn', knn_metric='minkowski', knne=False,
                 DSEL_perc=0.5, n_jobs=-1):
        self.pool_classifiers = pool_classifiers
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_samples)
            class labels of each example in X.

        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state_ = check_random_state(self.random_state)

        # Check if the pool of classifiers is None. If yes, use a
        # BaggingClassifier for the pool.
        if self.pool_classifiers is None:
            self.pool_classifiers_ = BaggingClassifier(
                random_state=self.random_state_, n_jobs=self.n_jobs)
            self.pool_classifiers_.fit(X, y)

        else:
            self.pool_classifiers_ = self.pool_classifiers

        self.n_classifiers_ = len(self.pool_classifiers_)
        # allow base models with feature subspaces.
        if hasattr(self.pool_classifiers_, "estimators_features_"):
            self.estimator_features_ = \
                np.array(self.pool_classifiers_.estimators_features_)
        else:
            indices = np.arange(X.shape[1])
            self.estimator_features_ = np.tile(indices,
                                               (self.n_classifiers_, 1))

        self._validate_pool()
        # dealing with label encoder
        self._check_label_encoder()
        self.y_enc_ = self._setup_label_encoder(y)
        self.n_classes_ = self.classes_.size
        self.n_features_ = X.shape[1]

        return self

    def _check_label_encoder(self):
        # Check if base classifiers are not using LabelEncoder (the case for
        # scikit-learn's ensembles):
        if isinstance(self.pool_classifiers_, BaseEnsemble):
            if np.array_equal(self.pool_classifiers_.classes_,
                              self.pool_classifiers_[0].classes_):
                self.base_already_encoded_ = False
            else:
                self.base_already_encoded_ = True
        else:
            self.base_already_encoded_ = False

    def _setup_label_encoder(self, y):
        """
        Setup the label encoder
        """
        self.enc_ = LabelEncoder()
        y_ind = self.enc_.fit_transform(y)
        self.classes_ = self.enc_.classes_

        return y_ind

    def _encode_base_labels(self, y):
        if self.base_already_encoded_:
            return y
        else:
            return self.enc_.transform(y)

    def _validate_pool(self):
        """ Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute.

        Raises
        -------
        ValueError
            If the pool of classifiers is empty or just a single model.
        """
        if self.n_classifiers_ <= 1:
            raise ValueError("n_classifiers must be greater than one, "
                             "got {}.".format(len(self.pool_classifiers)))

    def _predict_base(self, X):
        """ Get the predictions of each base classifier in the pool for all
            samples in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The test examples.

        Returns
        -------
        predictions : array of shape (n_samples, n_classifiers)
                      The predictions of each base classifier for all samples
                      in X.
        """
        predictions = np.zeros((X.shape[0], self.n_classifiers_),
                               dtype=np.intp)

        for index, clf in enumerate(self.pool_classifiers_):
            labels = clf.predict(X)
            predictions[:, index] = self._encode_base_labels(labels)
        return predictions

    def _preprocess_dsel(self):
        """Compute the prediction of each base classifier for
        all samples in DSEL. Used to speed-up the test phase, by
        not requiring to re-classify training samples during test.

        Returns
        -------
        DSEL_processed_ : array of shape (n_samples, n_classifiers).
                         Each element indicates whether the base classifier
                         predicted the correct label for the corresponding
                         sample (True), otherwise (False).

        BKS_DSEL_ : array of shape (n_samples, n_classifiers)
                   Predicted labels of each base classifier for all samples
                   in DSEL.
        """
        BKS_dsel = self._predict_base(self.DSEL_data_)
        processed_dsel = BKS_dsel == self.DSEL_target_[:, np.newaxis]

        return processed_dsel, BKS_dsel

    def _set_dsel(self, X, y):
        """Pre-Process the input X and y data into the dynamic selection
        dataset(DSEL) and get information about the structure of the data
        (e.g., n_classes, N_samples, classes)

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The Input data.

        y : array of shape (n_samples)
            class labels of each sample in X.
        """
        self.DSEL_data_ = X
        self.DSEL_target_ = y
        self.n_classes_ = self.classes_.size
        self.n_features_ = X.shape[1]
        self.n_samples_ = self.DSEL_target_.size
        self.DSEL_processed_, self.BKS_DSEL_ = self._preprocess_dsel()