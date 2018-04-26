from sklearn.model_selection import cross_val_score

from sklearn import svm
import numpy as np

from SVMClassifier import SVMClassifier


class SVMRBFClassifier(SVMClassifier):

    def __init__(self,x_train, y_train, x_test, y_test, x_val, y_val):
        super(SVMRBFClassifier, self).__init__(x_train, y_train, x_test, y_test, x_val, y_val)
        self.Cs = np.logspace(-2, 3, 10)
        self.gammas = np.logspace(-2, 1, 10)
        self.bs = np.arange(-6.0, 6.0, 0.5)
        self.roc_name_png = 'rbf_roc.png'
        self.val_score_png = 'rbf_cv.png'
        self.roc_name_png = 'rbf_roc.png'
        self.val_png_title = 'RBF Kernel CV scores'

    def gamma_validation(self):
        val_scores = []
        val_scores_std = []
        for g in self.gammas:
            self.clf.gamma = g
            scores = cross_val_score(self.clf, self.x_val, self.y_val, cv=5)
            val_scores.append(np.mean(scores))
            val_scores_std.append(np.std(scores))

        self.plot_cross_val(self.gammas, val_scores, val_scores_std, 'RBF Kernel CV scores ($\gamma$ param)',
                            'rbf_cv_gamma.png', 'Parameter $\gamma$')

    def train_classifier(self):
        self.clf = svm.SVC(kernel='rbf')
        self.gamma_validation()
        self.c_validation()
        self.clf.C = 10**.1
        self.clf.gamma = 0.1
        self.clf.fit(self.x_train, self.y_train)
        self.old_b = self.clf.intercept_[0]
