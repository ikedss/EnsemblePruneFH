import numpy as np
from deslib.static.des_fh import EnsemblePruneFH
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Setting up the random state to have consistent results
rng = np.random.RandomState(42)

# Generate a classification dataset
X, y = make_classification(n_samples=1000, random_state=rng)

# Apply PCA to reduce dimensionality to 2D or 3D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.33, random_state=rng)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5, random_state=rng)

# Create and train an ensemble of classifiers
classifiers = [RandomForestClassifier(n_estimators=10, random_state=rng) for _ in range(5)]
for clf in classifiers:
    clf.fit(X_train, y_train)


# Initialize the DS technique with the trained classifiers
fh = EnsemblePruneFH(pool_classifiers=classifiers, random_state=rng)

# Fitting the DES technique
fh.fit(X_dsel, y_dsel)

# Calculate classification accuracy of the technique
print('Evaluating DS technique:')
print('Classification accuracy EnsemblePruneFH: ', fh.score(X_test, y_test))

# Initialize the DS technique with the trained classifiers
fh.count_overlapping_hyperboxes()
# Print the number of hyperboxes
fh.print_number_of_hyperboxes()
# Visualize the hyperboxes
fh.visualize_hyperboxes(fh.DSEL_data_)
