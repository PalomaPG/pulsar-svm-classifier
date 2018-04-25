from sklearn.model_selection import cross_val_score

from sklearn import svm
import numpy as np

from SVMClassifier import SVMClassifier


class SVMPolyClassifier(SVMClassifier):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super(SVMPolyClassifier,self).__init__(x_train, y_train, x_test, y_test, x_val, y_val)
        self.Cs = np.logspace(-10, 0, 10)
        self.bs = np.arange(-6.0, 6.0, 0.5)
        self.n_s = np.arange(0, 20, 1, dtype=int)
        self.val_score_png = 'poly_cv.png'
        self.roc_name_png = 'poly_roc.png'
        self.val_png_title = 'Polynomial Kernel CV scores (C param.)'

    def degree_validation(self):
        val_scores = []
        val_scores_std = []
        for n in self.n_s:
            self.clf.degree = n
            scores = cross_val_score(self.clf, self.x_val, self.y_val, cv=5)
            val_scores.append(np.mean(scores))
            val_scores_std.append(np.std(scores))

        self.plot_cross_val(self.n_s, val_scores, val_scores_std, 'Poly Kernel CV scores (n param)',
                            'poly_cv_n.png', 'Parameter n', semilog=False)

    def train_classifier(self):
        self.clf = svm.SVC(kernel='poly')
        self.degree_validation()
        self.clf.degree=1
        self.c_validation()
        self.clf.C = 1
        self.clf.fit(self.x_train, self.y_train)
        self.old_b = self.clf.intercept_[0]
