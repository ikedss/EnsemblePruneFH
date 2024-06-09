import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from deslib.static import EnsemblePruneFH
from deslib.static import StaticSelection
from deslib.static import StackedClassifier

# Setting up the random state to have consistent results
rng = np.random.RandomState(42)

# Generate a classification dataset
X, y = make_classification(n_samples=1000, random_state=rng)

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rng)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5, random_state=rng)

# Create and train an ensemble of classifiers
classifiers = [RandomForestClassifier(n_estimators=10, random_state=rng) for _ in range(5)]
for clf in classifiers:
    clf.fit(X_train, y_train)

# Initialize the DS technique with the trained classifiers
fh = EnsemblePruneFH(pool_classifiers=classifiers, random_state=rng)
st = StaticSelection(random_state=rng)
sta = StackedClassifier(random_state=rng)
st2 = StaticSelection(pool_classifiers=classifiers, random_state=rng)
sta2 = StackedClassifier(pool_classifiers=classifiers, random_state=rng)

# Fitting the DES technique
fh.fit(X_dsel, y_dsel)
st.fit(X_dsel, y_dsel)
sta.fit(X_dsel, y_dsel)
st2.fit(X_dsel, y_dsel)
sta2.fit(X_dsel, y_dsel)

# Calculate classification accuracy of the technique
print('Evaluating DS technique:')
print('Classification accuracy EnsemblePruneFH: ', fh.score(X_test, y_test))
print('Classification accuracy StaticSelection: ', st.score(X_test, y_test))
print('Classification accuracy StaticSelection pool: ', st2.score(X_test, y_test))
print('Classification accuracy StackedClassifier: ', sta.score(X_test, y_test))
print('Classification accuracy StackedClassifier pool: ', sta2.score(X_test, y_test))