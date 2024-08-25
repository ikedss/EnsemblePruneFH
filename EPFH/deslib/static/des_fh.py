import numpy as np
from deslib.static.base import BaseStaticEnsemble
from deslib.util.fuzzy_hyperbox import Hyperbox

class DESFH(BaseStaticEnsemble):
    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30,
                 random_state=None, knn_classifier='knn', DSEL_perc=0.5, HyperBoxes=[], theta=0.05, mu=0.991,
                 mis_sample_based=True):
        self.theta = theta
        self.mu = mu
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []
        self.NO_hypeboxes = 0
        super(DESFH, self).__init__(pool_classifiers=pool_classifiers, with_IH=with_IH, safe_k=safe_k,
                                    IH_rate=IH_rate, mode='hybrid', random_state=random_state, DSEL_perc=DSEL_perc)

    def fit(self, X, y):
        super(DESFH, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")

        self.DSEL_data_ = X
        self.DSEL_target_ = y
        self.DSEL_processed_ = np.zeros((X.shape[0], self.n_classifiers_), dtype=bool)
        for classifier_index in range(self.n_classifiers_):
            predictions = self.pool_classifiers[classifier_index].predict(X)
            self.DSEL_processed_[:, classifier_index] = (predictions == y)

        for classifier_index in range(self.n_classifiers_):
            if self.mis_sample_based:
                MissSet_indexes = ~self.DSEL_processed_[:, classifier_index]
                self.setup_hyperboxes(MissSet_indexes, classifier_index)
            else:
                WellSet_indexes = self.DSEL_processed_[:, classifier_index]
                self.setup_hyperboxes(WellSet_indexes, classifier_index)

    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        boxes_classifier = np.zeros((len(self.HBoxes), 1))
        boxes_W = np.zeros((len(self.HBoxes), self.n_features_))
        boxes_V = np.zeros((len(self.HBoxes), self.n_features_))
        boxes_center = np.zeros((len(self.HBoxes), self.n_features_))
        competences_ = np.ones([len(query), self.n_classifiers_]) if self.mis_sample_based else np.zeros([len(query), self.n_classifiers_])

        for i in range(len(self.HBoxes)):
            boxes_classifier[i] = self.HBoxes[i].clsr
            boxes_W[i] = self.HBoxes[i].Max
            boxes_V[i] = self.HBoxes[i].Min
            boxes_center[i] = (self.HBoxes[i].Max + self.HBoxes[i].Min) / 2

        boxes_W = boxes_W.reshape(self.NO_hypeboxes, 1, self.n_features_)
        boxes_V = boxes_V.reshape(self.NO_hypeboxes, 1, self.n_features_)
        boxes_center = boxes_center.reshape(self.NO_hypeboxes, 1, self.n_features_)
        Xq = query.reshape(1, len(query), self.n_features_)

        halfsize = ((boxes_W - boxes_V) / 2).reshape(self.NO_hypeboxes, 1, self.n_features_)
        d = np.abs(boxes_center - Xq) - halfsize
        d[d < 0] = 0
        dd = np.linalg.norm(d, axis=2) / np.sqrt(self.n_features_)
        m = np.power(1 - dd, 4)

        classifiers, indices, count = np.unique(boxes_classifier, return_counts=True, return_index=True)
        for k, clsr in enumerate(classifiers):
            c_range = range(indices[k], indices[k] + count[k])
            cmat = m[c_range]
            if len(c_range) > 1:
                bb_indexes = np.argsort(-cmat, axis=0)
                b1, b2 = bb_indexes[0, :], bb_indexes[1, :]
                for i in range(len(query)):
                    competences_[i, int(clsr)] = cmat[b1[i], i] * 0.7 + cmat[b2[i], i] * 0.3
            else:
                for i in range(len(query)):
                    competences_[i, int(clsr)] = cmat[0, i]

        if self.mis_sample_based:
            competences_ = np.sqrt(self.n_features_) - competences_
        return competences_

    def setup_hyperboxes(self, samples_ind, classifier):
        if np.size(samples_ind) < 1:
            return
        boxes = []
        selected_samples = self.DSEL_data_[samples_ind, :]
        for X in selected_samples:
            if len(boxes) < 1:
                boxes.append(Hyperbox(v=X, w=X, classifier=classifier, theta=self.theta))
                self.NO_hypeboxes += 1
                continue

            if any(np.all(box.Min < X) and np.all(box.Max > X) for box in boxes):
                continue

            nearest_box = min(boxes, key=lambda box: np.linalg.norm(X - box.Center))
            if nearest_box.is_expandable(X):
                nearest_box.expand(X)
            else:
                boxes.append(Hyperbox(v=X, w=X, classifier=classifier, theta=self.theta))
                self.NO_hypeboxes += 1

        self.HBoxes.extend(boxes)

    def select(self, competences):
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)
        max_value = np.max(competences, axis=1)
        return competences >= self.mu * max_value.reshape(competences.shape[0], -1)

    def predict(self, X):
        predictions = np.asarray([clf.predict(X) for clf in self.pool_classifiers])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
