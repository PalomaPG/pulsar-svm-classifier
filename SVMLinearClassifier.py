from sklearn import svm
import numpy as np

from SVMClassifier import SVMClassifier


class SVMLinearClassifier(SVMClassifier):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super(SVMLinearClassifier, self).__init__(x_train, y_train, x_test, y_test, x_val, y_val)
        self.Cs = np.logspace(-10, 0, 10)
        self.bs = np.arange(-6.0, 6.0, 0.5)
        self.val_score_png = 'linear_cv.png'
        self.roc_name_png = 'linear_roc.png'
        self.val_png_title = 'Linear Model CV scores (C param.)'

    def train_classifier(self):
        self.clf = svm.SVC(kernel='linear')
        # self.c_validation()
        self.clf.C = 1
        self.clf.fit(self.x_train, self.y_train)
        self.old_b = self.clf.intercept_[0]



