from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import gdown

class MachineLearning():

    def __init__(self):
        print("Loading dataset ...")
        self.counter = 0

        self.flow_dataset = pd.read_csv('FlowStatsfile.csv')
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')

        self.X_flow = self.flow_dataset.iloc[:, :-1].values
        self.X_flow = self.X_flow.astype('float64')

        self.y_flow = self.flow_dataset.iloc[:, -1].values

        self.X_flow_train, self.X_flow_test, self.y_flow_train, self.y_flow_test = train_test_split(self.X_flow, self.y_flow, test_size=0.25, random_state=0)

    def LR(self):
        print("------------------------------------------------------------------------------")
        print("Logistic Regression ...")
        self.classifier = LogisticRegression(solver='liblinear', random_state=0)
        self.Confusion_matrix()

    def KNN(self):
        print("------------------------------------------------------------------------------")
        print("K-NEAREST NEIGHBORS ...")

        # اختيار عينة عشوائية من البيانات
        smaller_dataset = self.flow_dataset.sample(frac=0.1, random_state=1)

        X_flow = smaller_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = smaller_dataset.iloc[:, -1].values

        # استخدام PCA لتقليل الأبعاد
        pca = PCA(n_components=0.95)
        X_flow = pca.fit_transform(X_flow)

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        self.classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.Confusion_matrix(X_flow_train, X_flow_test, y_flow_train, y_flow_test)

    def SVM(self):
        """
        # Commenting out SVM
        print("------------------------------------------------------------------------------")
        print("SUPPORT-VECTOR MACHINE ...")
        self.classifier = SVC(kernel='rbf', random_state=0)
        self.Confusion_matrix()
        """

    def NB(self):
        print("------------------------------------------------------------------------------")
        print("NAIVE-BAYES ...")
        self.classifier = GaussianNB()
        self.Confusion_matrix()

    def DT(self):
        print("------------------------------------------------------------------------------")
        print("DECISION TREE ...")
        self.classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        self.Confusion_matrix()

    def RF(self):
        print("------------------------------------------------------------------------------")
        print("RANDOM FOREST ...")
        self.classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.Confusion_matrix()

    def Confusion_matrix(self, X_train=None, X_test=None, y_train=None, y_test=None):
        self.counter += 1

        if X_train is None:
            X_train = self.X_flow_train
            X_test = self.X_flow_test
            y_train = self.y_flow_train
            y_test = self.y_flow_test

        self.flow_model = self.classifier.fit(X_train, y_train)
        y_flow_pred = self.flow_model.predict(X_test)

        print("------------------------------------------------------------------------------")

        print("confusion matrix")
        cm = confusion_matrix(y_test, y_flow_pred)
        print(cm)

        acc = accuracy_score(y_test, y_flow_pred)

        print("success accuracy = {0:.2f} %".format(acc * 100))
        fail = 1.0 - acc
        print("fail accuracy = {0:.2f} %".format(fail * 100))
        print("------------------------------------------------------------------------------")

        x = ['TP', 'FP', 'FN', 'TN']
        x_indexes = np.arange(len(x))
        width = 0.10
        plt.xticks(ticks=x_indexes, labels=x)
        plt.title("Résultats des algorithmes")
        plt.xlabel('Classe predite')
        plt.ylabel('Nombre de flux')
        plt.tight_layout()
        plt.style.use("seaborn-darkgrid")

        if self.counter == 1:
            y1 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes - 2 * width, y1, width=width, color="#1b7021", label='LR')
            plt.legend()
        elif self.counter == 2:
            y2 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            # قبل رسم الرسم البياني، قم بضرب قيم طول عمود KNN بمعامل 10
            y2 = [val * 10 for val in y2]
            plt.bar(x_indexes - width, y2, width=width, color="#e46e6e", label='KNN')
            plt.legend()
        elif self.counter == 3:
            y3 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes, y3, width=width, color="#0000ff", label='NB')
            plt.legend()
        elif self.counter == 4:
            y4 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes + width, y4, width=width, color="#e0d692", label='DT')
            plt.legend()
        elif self.counter == 5:
            y5 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes + 2 * width, y5, width=width, color="#000000", label='RF')
            plt.legend()
            plt.show()

def main():
    start_script = datetime.now()
    ml = MachineLearning()

    start = datetime.now()
    ml.LR()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.KNN()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    # Commenting out SVM
    # start = datetime.now()
    # ml.SVM()
    # end = datetime.now()
    # print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.NB()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.DT()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.RF()
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    end_script = datetime.now()
    print("Script Time: ", (end_script - start_script))

if __name__ == "__main__":
    main()