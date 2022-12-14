import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.mixture import GaussianMixture

m1 = [-1, -1]
c1 = [[1,0], [0,1]]
m2 = [-1, 1]
c2 = [[0.7, 0.3], [0.3, 0.7]]
m3 = [1, -1]
c3 = [[0.8, 0.2], [0.2, 0.8]]
m4 = [1, 1]
c4 = [[0.5, 0.5], [0.5, 0.5]]
p_set = [0.1, 0.2, 0.3, 0.4]
def generate_dataset(k):
    which = np.random.choice(4, size=k, p=p_set)
    dataset = []
    for w in which:
        x = np.random.multivariate_normal(eval("m" + str(w + 1)), eval("c" + str(w + 1)))
        dataset.append(np.array([x, w]))
    return np.stack(dataset)

def log_likelihood_score(estimator, X, y=None):
    return estimator.score(X)
def exp_kfold_cv(k, data_s, p_set):
    data = generate_dataset(data_s)
    max_score = None
    best = p_set[0]
    for p in p_set:
        gmm = GaussianMixture(n_components=p)
        score = np.max(cross_validate(estimator=gmm, X=np.stack(data[:,0]),y=None, cv=k, scoring=log_likelihood_score)["test_score"])
        if max_score is None:
            max_score = score
        elif score > max_score:
            max_score = score
            best = p
    return best, score


def run_experiments(n):
    p_set = [1,2,3,4,5,6]
    data_sizes = [10, 100,1000,10000]
    for d in data_sizes:
        print("Performing experiments with {} samples".format(d))
        results_p = []
        scores = []
        for _ in range(n):
            best_p, score = exp_kfold_cv(10, d, p_set)
            results_p.append(best_p)
            scores.append(score)
        plt.clf()
        plt.hist(results_p, bins=[0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        plt.title("Best component for {} samples over {} experiments".format(d,n))
        plt.savefig("ps_{}.png".format(d))
        plt.clf()
        plt.plot(range(n), scores)
        plt.title("Cross Validation Likelihood \nover {} experiments for {} training samples".format(n, d))
        plt.savefig("loss_{}.png".format(d))

run_experiments(30)
# plt.clf()
# data = np.stack(generate_dataset(10000)[:,0])
# plt.contour(data[:,0], data[:,1])
# plt.savefig("data_vis.png")