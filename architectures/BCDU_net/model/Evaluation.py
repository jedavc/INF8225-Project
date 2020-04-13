from architectures.BCDU_net.model.Preprocessing import *
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

class Evaluation():
    def __init__(self):
        retina_blood_vessel_dataset = RetinaBloodVesselDataset()
        self.test_inputs, self.test_gt, self.test_bm = retina_blood_vessel_dataset.get_testing_data()
        self.evaluations_path = "../architectures/BCDU_net/Tests/"
        self.y_pred = None
        self.y_true = None
        self.jacc = None
        self.f1 = None
        self.roc = None
        self.prec_rec = None
        self.conf_matrix = None
        self.new_pred = None
        self.acc = None
        self.sens = None
        self.spec = None


    def evaluation_data(self):
        preprocessing = Preprocessing()
        test_prepro_inputs, test_prepro_bm = preprocessing.run_preprocess_pipeline(self.test_inputs, "test", self.test_gt)
        new_height, new_width = preprocessing.new_dimensions()
        test_prepro_inputs = np.einsum('klij->kijl', test_prepro_inputs)
        return test_prepro_inputs, test_prepro_bm, new_height, new_width

    def jaccard_metric(self):
        # Jaccard similarity index
        jc = jaccard_similarity_score(self.y_true, self.new_pred, normalize=True)
        print("\nJaccard similarity score: " + str(jc))
        return jc


    def f1_score(self):
        f1 = f1_score(self.y_true, self.new_pred, labels=None, average='binary', sample_weight=None)
        print("\nF1 score (F-measure): " + str(f1))
        return f1

    def ROC(self):
        false_positives, true_positives, thresholds = sklearn.metrics.roc_curve((self.y_true), self.y_pred)
        AUC_ROC = roc_auc_score(self.y_true, self.y_pred)
        # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
        print("\nArea under the ROC curve: " + str(AUC_ROC))
        roc_curve = plt.figure()
        plt.plot(false_positives, true_positives, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
        plt.title('ROC curve')
        plt.xlabel("FPR (False Positive Rate)")
        plt.ylabel("TPR (True Positive Rate)")
        plt.legend(loc="lower right")
        plt.savefig(self.evaluations_path + "ROC.png")
        return AUC_ROC

    def precision_recall_cuve(self):
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred)
        precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
        recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
        AUC_prec_rec = np.trapz(precision, recall)
        print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
        prec_rec_curve = plt.figure()
        plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
        plt.title('Precision - Recall curve')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower right")
        plt.savefig(self.evaluations_path +"Precision_recall.png")
        return AUC_prec_rec

    def confusion_matrix(self):
        threshold_confusion = 0.5
        print("\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
        new_pred = np.empty((self.y_pred.shape[0]))
        for i in range(self.y_pred.shape[0]):
            if self.y_pred[i] >= threshold_confusion:
                new_pred[i] = 1
            else:
                new_pred[i] = 0
        confusion = confusion_matrix(self.y_true, new_pred)
        print(confusion)
        self.new_pred = new_pred
        self.confusion_matrix = confusion

    def accuracy(self):
        sum = float(np.sum(self.confusion_matrix))
        accuracy = float(self.confusion_matrix[0,0] + self.confusion_matrix[1,1])/sum
        print("Accuracy: " + str(accuracy))
        return accuracy

    def specificity(self):
        denomin = float(self.confusion_matrix[0, 0] + self.confusion_matrix[0, 1])
        specificity = float(self.confusion_matrix[0, 0]) /denomin
        print("Specificity: " + str(specificity))
        return specificity

    def sensitivity(self):
        denom = float(self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0])
        sensitivity = float(self.confusion_matrix[1, 1]) /denom
        print("Sensitivity: " +str(sensitivity))
        return sensitivity

    def set_y_true_pred(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluation_metrics(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.conf_matrix = self.confusion_matrix()
        self.jacc = self.jaccard_metric()
        self.f1 = self.f1_score()
        self.roc = self.ROC()
        self.prec_rec = self.precision_recall_cuve()
        self.acc = self.accuracy()
        self.sens = self.sensitivity()
        self.spec = self.specificity()
        self.save_metrics_results()


    def save_metrics_results(self):
        # Save the results
        file_perf = open(self.evaluations_path + 'performances.txt', 'w')
        file_perf.write("Area under the ROC curve: " + str(self.roc)
                        + "\nArea under Precision-Recall curve: " + str(self.prec_rec)
                        + "\nJaccard similarity score: " + str(self.jacc)
                        + "\nF1 score (F-measure): " + str(self.f1)
                        + "\n\nConfusion matrix:"
                        + str(self.conf_matrix)
                        + "\nACCURACY: " + str(self.acc)
                        + "\nSENSITIVITY: " + str(self.sens)
                        + "\nSPECIFICITY: " + str(self.spec)
                        )
        file_perf.close()
