**Name : Samiksha Surawashi**

**Task 1 : Iris Flowers Classification ML Project**

**Importing Librarires**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""**Dataset**"""

from sklearn.datasets import load_iris

dataset = load_iris()

X = dataset.data 
y = dataset.target

X.shape

y.shape

"""**Visualizing dataset**"""

plt.plot(X[:, 0][y == 0] * X[:, 1][y == 0], X[:, 2][y == 0] * X[:, 3][y == 0], 'r.', label="Satosa")
plt.plot(X[:, 0][y == 1] * X[:, 1][y == 1], X[:, 2][y == 1] * X[:, 3][y == 1], 'g.', label="Virginica")
plt.plot(X[:, 0][y == 2] * X[:, 1][y == 2], X[:, 2][y == 2] * X[:, 3][y == 2], 'b.', label="Versicolour")
plt.legend()
plt.show()

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

"""**Logistic Regression**"""

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

"""**Check accuracy**"""

log_reg.score(X, y)

log_reg.score(X_train, y_train)

log_reg.score(X_test, y_test)
