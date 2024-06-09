import numpy as np
from deslib.static.base import BaseStaticEnsemble
from deslib.util.fuzzy_hyperbox import Hyperbox


class EnsemblePruneFH(BaseStaticEnsemble):
    def __init__(self, pool_classifiers=None, pct_classifiers=0.5, scoring=None, with_IH=False, safe_k=None,
                 IH_rate=0.30, random_state=None, DSEL_perc=0.5, HyperBoxes=[], theta=0.05, mu=0.991, n_jobs=-1,
                 mis_sample_based=True):
        self.theta = theta
        self.mu = mu
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []
        self.NO_hypeboxes = 0
        super(EnsemblePruneFH, self).__init__(pool_classifiers=pool_classifiers,
                                              with_IH=with_IH,
                                              safe_k=safe_k,
                                              IH_rate=IH_rate,
                                              mode='hybrid',  # hybrid, weighting
                                              random_state=random_state,
                                              DSEL_perc=DSEL_perc)
        if self.pool_classifiers is None:
            raise ValueError("pool_classifiers cannot be None")

    def fit(self, X, y):
        self.DSEL_data_ = X
        self.DSEL_target_ = y

        for clf in self.pool_classifiers:
            predictions = clf.predict(X)
            misclassified_indices = np.where(predictions != y)[0]
            self.setup_hyperboxes(misclassified_indices, clf)

    def setup_hyperboxes(self, samples_ind, classifier):
        if np.size(samples_ind) < 1:
            return
        boxes = []
        selected_samples = self.DSEL_data_[samples_ind, :]
        for X in selected_samples:
            if len(boxes) < 1:
                b = Hyperbox(v=X, w=X, classifier=classifier, theta=self.theta)
                self.NO_hypeboxes += 1
                boxes.append(b)
                continue
            IsInBox = False
            for box in boxes:
                if np.all(box.Min < X) and np.all(box.Max > X):
                    IsInBox = True
            if IsInBox:
                continue
            nDist = np.inf
            nearest_box = None
            for box in boxes:
                dist = np.linalg.norm(X - box.Center)
                if dist < nDist:
                    nearest_box = box
                    nDist = dist
            if nearest_box.is_expandable(X):
                nearest_box.expand(X)
                continue
            b = Hyperbox(v=X, w=X, classifier=classifier, theta=self.theta)
            boxes.append(b)
            self.NO_hypeboxes += 1
        self.HBoxes.extend(boxes)

    def predict(self, X):
        predictions = np.asarray([clf.predict(X) for clf in self.pool_classifiers])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote
