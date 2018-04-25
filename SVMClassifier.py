from sklearn.model_selection import cross_val_score


import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class SVMClassifier(ABC):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.Cs = None
        self.val_score_png = None
        self.roc_name_png = None
        self.clf = None
        self.val_png_title = None
        self.bs = []
        self.old_b = 0
        self.roc_curve = []

    @abstractmethod
    def train_classifier(self):
        pass

    def c_validation(self):
        val_scores = []
        val_scores_std = []
        for C in self.Cs:
            self.clf.C = C
            scores = cross_val_score(self.clf, self.x_val, self.y_val, cv=5)
            val_scores.append(np.mean(scores))
            val_scores_std.append(np.std(scores))

        self.plot_cross_val(self.Cs, val_scores, val_scores_std, self.val_png_title, self.val_score_png, 'Parameter C')

    def plot_cross_val(self, params_props, val_scores, val_scores_std,
                       val_png_title,  val_score_png, parameter, semilog=True):

        plt.figure()
        plt.clf()
        if semilog:
            plt.semilogx(params_props, val_scores)
            plt.semilogx(params_props, np.array(val_scores) + np.array(val_scores_std), 'b--')
            plt.semilogx(params_props, np.array(val_scores) - np.array(val_scores_std), 'b--')
        else:
            plt.plot(params_props, val_scores)
            plt.plot(params_props, np.array(val_scores) + np.array(val_scores_std), 'b--')
            plt.plot(params_props, np.array(val_scores) - np.array(val_scores_std), 'b--')
        locs, labels = plt.yticks()
        plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
        plt.ylabel('CV score')
        plt.xlabel(parameter)
        plt.title(val_png_title)
        plt.savefig(val_score_png)

    def get_score(self):
        self.train_classifier()
        score = self.clf.decision_function(self.x_test)
        return score

    def classify(self, b):
        w = np.array(self.get_score()+self.old_b-b, dtype=float)
        d = np.array([self.scale_answer(x) for x in w], dtype=float)
        return self.calc_rates(d)

    def calc_rates(self, d):
        tpr = 0  # true positive rate
        fn = 0   # false negatives
        fpr = 0  # false positive rate
        tn = 0   # true negatives
        for i in range(0, len(self.y_test)):
            if self.y_test[i]==d[i]:
                if d[i] == 0.0:
                    tn = tn + 1
                else:
                    tpr = tpr + 1
            else:
                if d[i] == 0.0:
                    fn = fn + 1
                else:
                    fpr = fpr + 1

        fpr = float(fpr)/float(fpr+tn)
        tpr = float(tpr)/float(tpr+fn)
        return [tpr, fpr]

    def get_roc_curve(self):

        for b in self.bs:
            self.roc_curve.append(self.classify(b))
        self.roc_curve = np.array(self.roc_curve, dtype='float64')
        self.plot_roc(self.roc_curve)
        return self.roc_curve

    def scale_answer(self, input):
        if input < 0:
            return 0.0
        else:
            return 1.0

    def plot_roc(self, roc_curve):
        plt.figure()
        lw = 2
        plt.plot(roc_curve[:, 1], roc_curve[:, 0], color='darkorange',
                 lw=lw, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.title('Receiver operating characteristic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(self.roc_name_png)
