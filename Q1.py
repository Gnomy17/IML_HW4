# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MLP(nn.Module):
#     def __init__(self, in_dim, out_dim, P):
#         super(MLP, self).__init__()
#         self.FC1 = nn.Linear(in_dim, P)
#         self.act1 = nn.ELU()
#         self.FC2 = nn.Linear(P, out_dim)
#         self.softmax = nn.Softmax()
    
#     def forward(self, x):
#         x = self.FC1(x)
#         x = self.act1(x)
#         x = self.FC2(x)
#         return self.softmax(x)

from scipy.stats import multivariate_normal as mvn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from generate_data import m1,c1, m2, c2, m3, c3, m4, c4
import warnings
warnings.filterwarnings('ignore') 

p1 = 0.25
p2 = 0.25
p3 = 0.25
p4 = 0.25
rv1 = mvn(mean=m1, cov = c1)
rv2 = mvn(mean=m2, cov = c2)
rv3 = mvn(mean=m3, cov = c3)
rv4 = mvn(mean=m4, cov = c4)
def make_decision(x):
    #Theoretically optimal classifier
    post1 = p1 * rv1.pdf(x)
    post2 = p2 * rv2.pdf(x)
    post3 = p3 * rv3.pdf(x)
    post4 = p4 * rv4.pdf(x)
    return np.argmax([post1, post2, post3, post4]) + 1

def perform_kcv(k, dataset, p_set):
    max_score=None
    best = p_set[0]
    for p in p_set:
        mlp = MLPClassifier(hidden_layer_sizes = (p,))
        score = np.max(cross_validate(mlp, np.stack(dataset[:,0]), dataset[:,1].astype('int'), cv=k, scoring="neg_log_loss")["test_score"])
        if max_score is None:
            max_score = score
        elif score > max_score:
            max_score = score
            best = p
    return best, max_score
test_d = np.load("test.npy", allow_pickle=True)
accs = []
l_set = [100,200,500,1000,2000,5000]
for le in l_set:
    d = np.load("training_{}.npy".format(le), allow_pickle=True)
    p, s = perform_kcv(10, d, [20, 100, 1000, 5000])
    print("Training set with {} samples has best p = {} with score {:.4f}.".format(le, p, s))
    print("Fitting data...")
    final_mlp = MLPClassifier(hidden_layer_sizes=(p,)).fit(np.stack(d[:,0]), d[:,1].astype('int'))
    print("Done.")
    preds = final_mlp.predict(np.stack(test_d[:,0]))
    acc = accuracy_score(preds, test_d[:,1].astype('int'))
    accs.append(acc)
    plt.scatter(le, 1 - acc, label="p = {}".format(p))

preds_opt = np.stack(list(map(make_decision, test_d[:,0])))
acc_opt = accuracy_score(preds_opt, test_d[:,1].astype('int'))
plt.plot(l_set, [1- acc_opt for l in l_set], label="opt class err")
plt.legend()
plt.savefig("things.png", dpi=400)

