from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plot

def decisionTree(train_x,train_y,test_x,test_y):
    print("Using Decision trees:")
    clf = DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    predcted_y = clf.predict(test_x)
    accuracy = clf.score(test_x, test_y)
    print(f"Accuracy: {accuracy}")
    f_measure = f1_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"F_measure: {f_measure}")
    recall = recall_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Recall: {recall}")
    precision = precision_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Precision: {precision}\n\n")
    # plot_confusion_matrix(clf, test_x, test_y)
    # plot.show()

def randomForest(train_x,train_y,test_x,test_y,n_estimators=100):
    print(f"Using random forest, N_estimators: {n_estimators}")
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(train_x, train_y)
    predcted_y = clf.predict(test_x)
    accuracy = clf.score(test_x, test_y)
    print(f"Accuracy: {accuracy}")
    f_measure = f1_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"F_measure: {f_measure}")
    recall = recall_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Recall: {recall}")
    precision = precision_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Precision: {precision}\n\n")
    # plot_confusion_matrix(clf, test_x, test_y)
    # plot.show()

def adaBoost(train_x,train_y,test_x,test_y,n_estimators=100):
    print(f"Using AdaBoost, N_estimators: {n_estimators}")
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    clf.fit(train_x, train_y)
    predcted_y = clf.predict(test_x)
    accuracy = clf.score(test_x, test_y)
    print(f"Accuracy: {accuracy}")
    f_measure = f1_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"F_measure: {f_measure}")
    recall = recall_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Recall: {recall}")
    precision = precision_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Precision: {precision}\n\n")
    # plot_confusion_matrix(clf, test_x, test_y)
    # plot.show()

def kNearestNeighbor(train_x,train_y,test_x,test_y,k=3):
    print(f"Using K Nearest Neighbour, K: {k}")
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_x, train_y)
    predcted_y = clf.predict(test_x)
    accuracy = clf.score(test_x, test_y)
    print(f"Accuracy: {accuracy}")
    f_measure = f1_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"F_measure: {f_measure}")
    recall = recall_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Recall: {recall}")
    precision = precision_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Precision: {precision}\n\n")


def naiveBayes(train_x,train_y,test_x,test_y):
    clf = GaussianNB()
    clf.fit(train_x, train_y)
    predcted_y = clf.predict(test_x)
    print("Using Naive Bayes:")
    accuracy = clf.score(test_x, test_y)
    print(f"Accuracy: {accuracy}")
    f_measure = f1_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"F_measure: {f_measure}")
    recall = recall_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Recall: {recall}")
    precision = precision_score(test_y, predcted_y, average=None, labels=['g', 'h'])
    print(f"Precision: {precision}\n\n")
