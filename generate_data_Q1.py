import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt


m1 = [0,0,2*np.sqrt(2/3)]
m2 = [-1, -2*np.sqrt(3)/6, 0]
m3 = [1, -2*np.sqrt(3)/6, 0]
m4 = [0,2*np.sqrt(3)/3, 0]
c1 = [[1,0,0],[0,1,0], [0,0,1]]
c2 = [[1,0,0],[0,1,0], [0,0,1]]
c3 = [[1,0,0],[0,1,0], [0,0,1]]
c4 = [[1,0,0],[0,1,0], [0,0,1]]
def sample_xc1():
    return np.random.multivariate_normal(m1, c1)
def sample_xc2():
    return np.random.multivariate_normal(m2, c2)
def sample_xc3():
    return np.random.multivariate_normal(m3, c3)
def sample_xc4():
    return np.random.multivariate_normal(m4, c4)
def generate_dataset(num_samps, name):
    labels = np.random.choice([1,2,3,4], num_samps, p=[0.25, 0.25, 0.25, 0.25])
    samples = []
    for l in labels:
        func = "sample_xc" + str(l) + "()"
        samples.append(eval(func))
    dataset = np.array(list(zip(np.stack(samples), labels)))
    np.save(name + ".npy", dataset)

for d in [100,200,500,1000,2000,5000]:
    generate_dataset(d, "training_" + str(d))

generate_dataset(100000, "test")
