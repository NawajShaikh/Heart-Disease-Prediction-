# imorting heart data
import pandas as pd
heart = pd.read_csv("D:\\Kaggle\\heart-disease-uci\\heart.csv")
heart.shape     #shape of the data
heart.columns   # all column names in the data
heart.head()    # first 5 rows of data (how data looks like)
heart.isnull().sum()    # checking missing values in the data
heart.describe()    # summary of the data
heart.target.value_counts() # cheching data is balanced or not

import matplotlib.pyplot as plt
plt.hist(heart.target);plt.xlabel("Target");plt.ylabel("count")

# splitting the data in training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        heart.iloc[:,0:13], heart.target, random_state = 0)


# K-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
knn  = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(knn.score(X_train, y_train)))

knn_train_pred = knn.predict(X_train)
import numpy as np
knn_train_acc = np.mean(knn_train_pred == y_train)
knn_train_acc

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_train, knn_train_pred)
classification_report(y_train, knn_train_pred)

print("Accuracy on training set: {:.3f}".format(knn.score(X_test, y_test)))

knn_test_pred = knn.predict(X_test)
knn_test_acc = np.mean(knn_test_pred == y_test)
knn_test_acc

confusion_matrix(y_test, knn_test_pred)
classification_report(y_test, knn_test_pred)


# creating empty list variable 
acc = []
# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
for i in range(3,50,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    knn_train_acc = np.mean(knn.predict(X_train)==y_train)
    knn_test_acc = np.mean(knn.predict(X_test)==y_test)
    acc.append([knn_train_acc,knn_test_acc])


# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")
plt.legend(["train","test"])


# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression().fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(logreg.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(logreg001.score(X_test, y_test)))


plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
plt.xticks(range(heart.iloc[:,0:13].shape[1]), heart.iloc[:,0:13].columns, rotation=90)
plt.hlines(0, 0, heart.iloc[:,0:13].shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
            C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
            C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
plt.xticks(range(heart.iloc[:,0:13].shape[1]), heart.columns, rotation=90)
plt.hlines(0, 0, heart.iloc[:,0:13].shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)


# Deciion Tree

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


tree12 = DecisionTreeClassifier(max_depth=12, random_state=0)
tree12.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree12.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree12.score(X_test, y_test)))


import numpy as np
import matplotlib.pyplot as plt
def plot_feature_importances_heart(model):
    n_features = heart.iloc[:,0:13].shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), heart.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
plot_feature_importances_heart(tree)


# Random Forest

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

def plot_feature_importances_heart(model):
    n_features = heart.iloc[:,0:13].shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), heart.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
plot_feature_importances_heart(forest)


# Gradient boosted regression trees (gradient boosting machines)

from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt1 = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt1.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt1.score(X_test, y_test)))

gbrt001 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt001.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt001.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt001.score(X_test, y_test)))

def plot_feature_importances_heart(model):
    n_features = heart.iloc[:,0:13].shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), heart.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
plot_feature_importances_heart(gbrt)


# Linear SVC (Support Vector Classifier)

from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X_train, y_train)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)


# Kernelize Support Vector Machines

from sklearn.svm import SVC 
svc_rbf = SVC(kernel='rbf', C=10, gamma=0.1).fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(svc_rbf.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc_rbf.score(X_test, y_test)))

svc = SVC()
svc.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")

# compute the minimum value per feature on the training set
min_on_training = X_train.min(axis=0)
# compute the range of each feature (max - min) on the training set
range_on_training = (X_train - min_on_training).max(axis=0)
# subtract the min, and divide by range
# afterward, min=0 and max=1 for each feature
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
print("Maximum for each feature\n {}".format(X_train_scaled.max(axis=0)))

# use THE SAME transformation on the test set,
# using min and range of the training set (see Chapter 3 for details)
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC()
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))


# Neural Network (Deep Learning)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0).fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

mlp = MLPClassifier(random_state=0, hidden_layer_sizes=[10])
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

# using two hidden layers, with 10 units each
mlp = MLPClassifier(random_state=0,
                    hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

# using two hidden layers, with 10 units each, now with tanh nonlinearity
mlp = MLPClassifier(activation='tanh',
                    random_state=0, hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))


# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
# compute the standard deviation of each feature on the training set
std_on_train = X_train.std(axis=0)
# subtract the mean, and scale by inverse standard deviation
# afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
# use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
        mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))


mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
        mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))


mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
        mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))












































































































































































































































































































































































































































