**Name : Samiksha Surawashi**

**Task2 : Prediction using Decision Tree  Algorithm**

**Importing all required libraries**
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import tree
import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pydotplus

"""**Loading the iris dataset**"""

iris = datasets.load_iris()

"""**Forming the iris dataframe**"""

data = pd.DataFrame(iris.data, columns = iris.feature_names)
y = iris.target
y = pd.DataFrame(y,columns=["Target"])

df = pd.concat([data,y],axis=1)

df.head()

df.tail()

"""**Printing no. of rows and columns**"""

df.shape

"""**Columns Names in the Dataset**"""

df.columns

df.info()

df.describe()

df.dtypes

"""**Data Visualization**"""

sns.pairplot(df,hue="Target")

sns.distplot(df["sepal length (cm)"])

sns.distplot(df["sepal width (cm)"])

sns.distplot(df["petal length (cm)"])

sns.distplot(df["petal width (cm)"])

sns.pairplot(df,diag_kind='kde')

plt.figure(figsize=[10,8])
df.plot(kind="box",subplots=True,sharey=False,sharex=False,layout=(2,3))
plt.xticks(rotation=90)
plt.show()
plt.tight_layout()

plt.figure(figsize=[10,8])
sns.heatmap(df.corr(),annot=True) # To check correlation.
plt.title("Heatmap")
plt.show()

"""**Splitting data in x and y (Attributes and Labels Respectively)**"""

x = df.drop("Target",axis=1)
y = df.Target

"""**Performing a Train-Test Split (Splitting the data into training and testing sets)**"""

xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=123,test_size=0.2)

"""**Initializing the Decision Tree Model and Fitting the Data**"""

clf_tree=DecisionTreeClassifier(random_state=123)
dt_fit = clf_tree.fit(xtrain,ytrain)

"""**Prediction of the Decision Tree Model**"""

dt_predict = dt_fit.predict(xtest)

"""**Evaluation of the ML Model**"""

sns.heatmap(confusion_matrix(ytest,dt_predict),annot=True)

classificationreport = classification_report(ytest,dt_predict)
print("Classification Report is: ",classificationreport)

accuracy = accuracy_score(ytest,dt_predict)
print("Accuracy Score is:",accuracy)

"""**Visualize the graph**"""

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,10), dpi=300)
tree.plot_tree(clf_tree,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('/content/iris.png')

"""**THANK YOU**"""
