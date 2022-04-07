import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from kale.pipeline.multi_domain_adapter import _CoIRLS


def main():

    np.random.seed(81192)
    # Generate toy data
    n_samples = 1000

    xs, ys = make_blobs(n_samples, centers=[[0, 0], [0, 2]], cluster_std=[0.3, 0.35])
    xt, yt = make_blobs(n_samples, centers=[[2, -2], [2, 0.2]], cluster_std=[0.35, 0.4])

    # visualize toy data
    colors = ["c", "m"]
    x_all = [xs, xt]
    y_all = [ys, yt]
    labels = ["source", "Target"]
    plt.figure(figsize=(10, 5))
    for i in range(2):
        idx_pos = np.where(y_all[i] == 1)
        idx_neg = np.where(y_all[i] == 0)
        plt.scatter(
            x_all[i][idx_pos, 0],
            x_all[i][idx_pos, 1],
            c=colors[i],
            marker="o",
            alpha=0.4,
            label=labels[i] + " positive",
        )
        plt.scatter(
            x_all[i][idx_neg, 0],
            x_all[i][idx_neg, 1],
            c=colors[i],
            marker="x",
            alpha=0.4,
            label=labels[i] + " negative",
        )
    plt.legend()
    plt.title("Source domain and target domain blobs data", fontsize=14, fontweight="bold")
    plt.show()

    clf = RidgeClassifier(alpha=1.0)
    clf.fit(xs, ys)

    yt_pred = clf.predict(xt)
    print("Accuracy on target domain: {:.2f}".format(accuracy_score(yt, yt_pred)))

    # visualize decision scores of non-adaptation classifier
    ys_score = clf.decision_function(xs)
    yt_score = clf.decision_function(xt)
    plt.figure(figsize=(10, 5))
    sns.histplot(ys_score, color="c", label="Source", kde=True)
    sns.histplot(yt_score, color="m", label="Target", kde=True)
    plt.xlabel("Decision Scores")
    plt.legend()
    plt.title("Ridge classifier decision score distribution", fontsize=14, fontweight="bold")
    plt.show()

    # domain adaptation
    clf_ = _CoIRLS()
    # encoding one-hot domain covariate matrix
    covariates = np.zeros(n_samples * 2)
    covariates[:n_samples] = 1
    enc = OneHotEncoder(handle_unknown="ignore")
    covariates_mat = enc.fit_transform(covariates.reshape(-1, 1)).toarray()

    x = np.concatenate((xs, xt))
    clf_.fit(x, ys, covariates_mat)
    yt_pred_ = clf_.predict(xt)
    print("Accuracy on target domain: {:.2f}".format(accuracy_score(yt, yt_pred_)))

    ys_score_ = clf_.decision_function(xs).detach().numpy().reshape(-1)
    yt_score_ = clf_.decision_function(xt).detach().numpy().reshape(-1)
    plt.figure(figsize=(10, 5))
    sns.histplot(ys_score_, color="c", label="Source", kde=True)
    sns.histplot(yt_score_, color="m", label="Target", kde=True)
    plt.xlabel("Decision Scores")
    plt.legend()
    plt.title("Domain adaptation classifier decision score distribution", fontsize=14, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    main()
