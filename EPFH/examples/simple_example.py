import numpy as np
from deslib.static.epfh import EnsemblePruneFH
from deslib.static.des_fh import DESFH
from deslib.static.static_selection import StaticSelection
from deslib.static.stacked import StackedClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
import time

rng = np.random.RandomState(42)
X, y = make_classification(n_samples=5000, random_state=rng)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.33, random_state=rng)
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5, random_state=rng)

#classifiers = [RandomForestClassifier(n_estimators=50, random_state=rng) for _ in range(50)]
#for clf in classifiers:
#    clf.fit(X_train, y_train)
classifiers = BaggingClassifier(n_estimators=50, random_state=rng)
startBaggingClassifier = time.time()
classifiers.fit(X_train, y_train)
endBaggingClassifier = time.time()

#sc = StackedClassifier(pool_classifiers=classifiers, random_state=rng)
#startStackedClassifier = time.time()
#sc.fit(X_dsel, y_dsel)
#endStackedClassifier = time.time()

ss = StaticSelection(pool_classifiers=classifiers, random_state=rng)
startStaticSelection = time.time()
ss.fit(X_dsel, y_dsel)
endStaticSelection = time.time()

des = DESFH(pool_classifiers=classifiers, random_state=rng)
startDESFH = time.time()
des.fit(X_dsel, y_dsel)
endDESFH = time.time()

fh = EnsemblePruneFH(pool_classifiers=classifiers, random_state=rng)
startPrune = time.time()
fh.fit(X_dsel, y_dsel)
endPrune = time.time()

print('Evaluating DS technique:')
print('Classification accuracy BaggingClassifier: ', classifiers.score(X_test, y_test))
print('Time Bagging: ', endBaggingClassifier - startBaggingClassifier)

#print('Classification accuracy StackedClassifier: ', sc.score(X_test, y_test))
#print('Time StackedClassifier: ', endStackedClassifier - startStackedClassifier)

print('Classification accuracy StaticSelection: ', ss.score(X_test, y_test))
print('Time StaticSelection: ', endStaticSelection - startStaticSelection)

print('Classification accuracy DESFH: ', des.score(X_test, y_test))
print('Time DESFH: ', endDESFH - startDESFH)

print('Classification accuracy EnsemblePruneFH: ', fh.score(X_test, y_test))
print('Time EnsemblePruneFH: ', endPrune - startPrune)
