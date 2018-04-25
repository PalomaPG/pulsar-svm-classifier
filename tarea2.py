import sys
import numpy as np
from SVMLinearClassifier import SVMLinearClassifier
from SVMRBFClassifier import SVMRBFClassifier
from SVMPolyClassifier import SVMPolyClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def tarea2(path):
    data = np.genfromtxt(path, dtype=float, delimiter=',')
    np.random.shuffle(data)
    data_len = len(data)
    idx = int(data_len * .6)
    train_set = data[:idx]
    val_set = data[idx:(int(data_len * .2 + idx) + 1)]
    idx = int(data_len * .2 + idx) + 1
    test_set = data[idx:]

    scaler = StandardScaler().fit(train_set[:, :8], train_set[:, 8])
    x_train = scaler.transform(train_set[:, :8])
    y_train = train_set[:, 8]
    x_test = scaler.transform(test_set[:, :8])
    y_test = test_set[:, 8]
    x_val = scaler.transform(val_set[:, :8])
    y_val = val_set[:, 8]

    lin_svm = SVMLinearClassifier(x_train, y_train, x_test, y_test, x_val, y_val)
    lin_svm .train_classifier()
    lin_roc = lin_svm.get_roc_curve()

    rbf_svm = SVMRBFClassifier(x_train, y_train, x_test, y_test, x_val, y_val)
    rbf_svm.train_classifier()
    rbf_roc = rbf_svm.get_roc_curve()

    poly_svm = SVMPolyClassifier(x_train, y_train, x_test, y_test, x_val, y_val)
    poly_svm.train_classifier()
    poly_roc = poly_svm.get_roc_curve()

    plot_rocs(lin_roc, rbf_roc, poly_roc)


def plot_rocs(lin_roc, rbf_roc, poly_roc):
    plt.figure()
    lw = 2
    plt.plot(lin_roc[:, 1], lin_roc[:, 0], color='darkgreen',
             lw=lw, label='Linear Kernel')
    plt.plot(rbf_roc[:, 1], rbf_roc[:, 0], color='darkred',
             lw=lw, label='RBF Kernel')
    plt.plot(poly_roc[:, 1], poly_roc[:, 0], color='darkcyan',
             lw=lw, label='Poly Kernel')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.title('Receiver operating characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('all_rocs.png')


if __name__ == '__main__':
    tarea2(sys.argv[1])