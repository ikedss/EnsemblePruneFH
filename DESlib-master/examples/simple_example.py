import numpy as np
from deslib.static.des_fh import EnsemblePruneFH
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

rng = np.random.RandomState(42)
X, y = make_classification(n_samples=1000, random_state=rng)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.33, random_state=rng)
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5, random_state=rng)

classifiers = [RandomForestClassifier(n_estimators=10, random_state=rng) for _ in range(5)]
for clf in classifiers:
    clf.fit(X_train, y_train)

fh = EnsemblePruneFH(pool_classifiers=classifiers, random_state=rng)
fh.fit(X_dsel, y_dsel)
fh.count_overlapping_hyperboxes()
fh.print_number_of_hyperboxes()
fh.visualize_hyperboxes(fh.DSEL_data_)

print('Evaluating DS technique:')
print('Classification accuracy EnsemblePruneFH: ', fh.score(X_test, y_test))
