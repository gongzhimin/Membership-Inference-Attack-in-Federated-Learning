import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def set_matplotlib_font(font_weight="bold", font_size=10):
    font = {
        "weight": font_weight,
        "size": font_size
    }
    matplotlib.rc("font", **font)


class Visualizer:
    def __init__(self, filepath="logs/plots/", font_weight="bold", font_size=10):
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        self.filepath = filepath
        set_matplotlib_font(font_weight, font_size)
        self.membership_probability_histogram = None
        self.membership_inference_attack_roc_curve = None
        self.unique_labels = None
        self.gradient_norm_scatter = None
        self.per_label_membership_probability_histograms = []

    def plot_membership_probability_histogram(self, member_preds, nonmember_preds):
        self.membership_probability_histogram = plt.figure("membership probability histogram")

        if member_preds and len(member_preds):
            plt.hist(np.array(member_preds).flatten(),
                     color="xkcd:blue", alpha=0.7,
                     bins=20, histtype="bar", range=(0, 1),
                     weights=(np.ones_like(member_preds) / len(member_preds)),
                     label="Training Data (Members)")
        if nonmember_preds and len(nonmember_preds):
            plt.hist(np.array(nonmember_preds).flatten(),
                     color="xkcd:light blue", alpha=0.7,
                     bins=20, histtype="bar", range=(0, 1),
                     weights=(np.ones_like(nonmember_preds) / len(nonmember_preds)),
                     label="Population Data (Non-members)")
        plt.legend(loc="upper left")

        plt.ylabel("Fraction")
        plt.xlabel("Membership probability")
        plt.title("Privacy Risk")

        plt.savefig("{}membership_probability_histogram.svg".format(self.filepath))
        plt.close()

    def plot_membership_inference_attack_roc_curve(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        area_under_roc = auc(fpr, tpr)

        self.membership_inference_attack_roc_curve = plt.figure("membership inference attack roc curve")

        plt.plot([0, 1], [0, 1], "r--")
        plt.plot(fpr, tpr, "b", label="AUC={:.2f}".format(area_under_roc))
        plt.legend(loc="lower right")

        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.title("ROC of Membership Inference Attack")

        plt.savefig("{}membership_inference_attack_roc_curve.svg".format(self.filepath))
        plt.close()

    def plot_gradient_norm_scatter(self,
                                   member_labels, member_gradient_norms,
                                   nonmember_labels, nonmember_gradient_norms):
        xs = []
        member_ys, nonmember_ys = [], []
        self.unique_labels = sorted(np.unique(member_labels))
        for label in self.unique_labels:
            member_gradient_norm_list = []
            for (member_label, member_gradient_norm) in zip(member_labels, member_gradient_norms):
                if member_label == label:
                    member_gradient_norm_list.append(member_gradient_norm)

            nonmember_gradient_norm_list = []
            for (nonmember_label, nonmember_gradient_norm) in zip(nonmember_labels, nonmember_gradient_norms):
                if nonmember_label == label:
                    nonmember_gradient_norm_list.append(nonmember_gradient_norm)

            xs.append(label)
            member_ys.append(np.mean(member_gradient_norm_list))
            nonmember_ys.append(np.mean(nonmember_gradient_norm_list))

        self.gradient_norm_scatter = plt.figure("gradient norm scatter")

        plt.plot(xs, member_ys, "g.", label="Training Data (Members)")
        plt.plot(xs, nonmember_ys, "r.", label="Population Data (Non-members)")
        plt.legend(loc="upper left")

        plt.xlabel("Label")
        plt.ylabel("Average Gradient Norm")
        plt.title("Average Gradient Norms per Label")

        plt.savefig("{}gradient_norm_scatter.svg".format(self.filepath))
        plt.close()

    def plot_per_label_membership_probability_histogram(self,
                                                        member_labels, member_preds,
                                                        nonmember_labels, nonmember_preds):
        for label in self.unique_labels:
            member_preds_per_label = []
            for (member_label, member_pred) in zip(member_labels, member_preds):
                if member_label == label:
                    member_preds_per_label.append(member_pred)

            nonmember_preds_per_label = []
            for (nonmember_label, nonmember_pred) in zip(nonmember_labels, nonmember_preds):
                if nonmember_label == label:
                    nonmember_preds_per_label.append(nonmember_pred)

            per_label_histogram = plt.figure("label {} membership probability histogram".format(int(label)))
            self.per_label_membership_probability_histograms.append(per_label_histogram)

            if member_preds_per_label and len(member_preds_per_label):
                plt.hist(np.array(member_preds_per_label).flatten(),
                         color="xkcd:blue", alpha=0.7,
                         bins=20, histtype="bar", range=(0, 1),
                         weights=(np.ones_like(member_preds_per_label) / len(member_preds_per_label)),
                         label="Training Data (Members)")
            if nonmember_preds_per_label and len(nonmember_preds_per_label):
                plt.hist(np.array(nonmember_preds_per_label).flatten(),
                         color="xkcd:light blue", alpha=0.7,
                         bins=20, histtype="bar", range=(0, 1),
                         weights=(np.ones_like(nonmember_preds_per_label) / len(nonmember_preds_per_label)),
                         label="Population Data (Non-Members)")
            plt.legend(loc="upper left")

            plt.ylabel("Fraction")
            plt.xlabel("Membership Probability")
            plt.title("Privacy Risk - Label {}".format(int(label)))

            plt.savefig("{}membership_probability_histogram_label{}.svg".format(self.filepath, int(label)))
            plt.close()
