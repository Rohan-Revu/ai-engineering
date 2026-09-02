from sklearn.neighbors import NearestCentroid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# First: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Second: separate validation from remaining data
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)

clf = NearestCentroid()

# Learn ONLY from training data
clf.fit(X_train, y_train)

print(f"Training accuracy:   {clf.score(X_train, y_train):.3f}")
print(f"Validation accuracy: {clf.score(X_val, y_val):.3f}")
print(f"Test accuracy:       {clf.score(X_test, y_test):.3f}")