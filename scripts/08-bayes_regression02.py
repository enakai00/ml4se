import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import normal, multivariate_normal

beta = 1.0/(0.3)**2
alpha = 1.0/100**2
order = 9

def create_dataset(num):
    dataset = DataFrame(columns=['x','y'])
    for i in range(num):
        x = float(i)/float(num-1)
        y = np.sin(2.0*np.pi*x) + normal(scale=0.3)
        dataset = dataset.append(Series([x,y], index=['x','y']),
                                 ignore_index=True)
    return dataset

def resolve(dataset, m):
    t = dataset.y
    phis = DataFrame()
    for i in range(0,m+1):
        p = dataset.x**i
        p.name="x**%d" % i
        phis = pd.concat([phis,p], axis=1)

    for index, line in phis.iterrows():
        phi = DataFrame(line)
        if index == 0:
            phiphi = np.dot(phi,phi.T)
        else:
            phiphi += np.dot(phi,phi.T)
    s_inv = alpha * DataFrame(np.identity(m+1)) + beta * phiphi
    s = np.linalg.inv(s_inv)

    def mean_fun(x0):
        phi_x0 = DataFrame([x0 ** i for i in range(0,m+1)])
        for index, line in phis.iterrows():
            if index == 0:
                tmp = t[index] * line
            else:
                tmp += t[index] * line
        return (beta * np.dot(np.dot(phi_x0.T, s), DataFrame(tmp))).flatten()

    def deviation_fun(x0):
        phi_x0 = DataFrame([x0 ** i for i in range(0,m+1)])
        deviation = np.sqrt(1.0/beta + np.dot(np.dot(phi_x0.T, s), phi_x0))
        return deviation.diagonal()

    for index, line in phis.iterrows():
        if index == 0:
            tmp = t[index] * line
        else:
            tmp += t[index] * line
    mean = beta * np.dot(s, DataFrame(tmp)).flatten()

    return mean_fun, deviation_fun, mean, s

if __name__ == '__main__':
    df_ws = DataFrame()

    # Show fitting curves
    fig = plt.figure()
    for c, num in enumerate([4,5,10,100]): # Num of datapoints
        train_set = create_dataset(num)
        mean_fun, deviation_fun, mean, sigma = resolve(train_set, order)
        subplot = fig.add_subplot(2,2,c+1)
        subplot.set_xlim(-0.05,1.05)
        subplot.set_ylim(-2,2)
        subplot.set_title("N=%d" % num)
        ws_samples = DataFrame(multivariate_normal(mean,sigma,4))

        # dataset
        subplot.scatter(train_set.x, train_set.y, marker='o', color='blue')

        # correct curve
        linex = np.arange(0,1.01,0.01)

        # polynomial fit
        m = np.array(mean_fun(linex))
        d = np.array(deviation_fun(linex))
        liney = m
        subplot.plot(linex, liney, color='red', label="mean")
        subplot.legend(loc=1)

        def f(x, ws):
            y = 0
            for i, w in enumerate(ws):
                y += w * (x ** i)
            return y

        for index, ws in ws_samples.iterrows():
            liney = f(linex, ws)
            subplot.plot(linex, liney, color='red', linestyle='--')

    fig.show()
