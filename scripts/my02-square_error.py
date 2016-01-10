import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import normal

N=10
M=[0, 1, 3, 9]

def create_dataset(num):
    """
    create dataset {x_n, y_n}(n=1...N)
    """
    dataset = DataFrame(columns=['x', 'y'])
    for i in range(num):
        x = i / (num - 1)
        # noize with standard deviation 0.3
        y = np.sin(2 * np.pi * x) + normal(scale=0.3)
        dataset = dataset.append(Series([x,y], index=['x', 'y']),
                                 ignore_index=True)
    return dataset

def rmse(dataset, f):
    """
    Calicurate Root Mean Square Error
    """
    err = 0.
    for index, line in dataset.iterrows():
        x, y = line.x, line.y
        # sigma
        err += 0.5 * (y - f(x))**2
    return np.sqrt(2 * err / len(dataset))


def resolve(dataset, m):
    t = dataset.y
    phi = DataFrame()
    for i in range(0, m + 1):
        p = dataset.x ** i
        p.name="x**%d" % i
        # pd.concat(
        # Concatenate pandas object along a particular axis
        phi = pd.concat([phi, p], axis=1)
    # np.linalg.inv
    # compute the inverse of a matrix
    #
    # np.dot
    # Dot product of two arrays
    # For 2-D arrays it is equivalent to matrix multiplication, and for 1-D
    # arrays to inner product of verctors (without complex conjugation). For
    # N dimensions it is a sum product over the last axis of a and the second-to-
    # last of b::
    tmp = np.linalg.inv(np.dot(phi.T, phi))
    ws = np.dot(np.dot(tmp, phi.T), t)

    def f(x):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    return f, ws

def main():
    train_set = create_dataset(N)
    test_set = create_dataset(N)
    df_ws = DataFrame()

    fig = plt.figure()
    for c, m in enumerate(M):
        f, ws = resolve(train_set, m)
        df_ws = df_ws.append(Series(ws, name="M=%d" % m))

        subplot = fig.add_subplot(2, 2, c + 1)
        subplot.set_xlim(-0.05, 1.05)
        subplot.set_ylim(-1.5, 1.5)
        subplot.set_title("M=%d" % m)

        subplot.scatter(train_set.x, train_set.y, marker='o', color='blue')

        linex = np.linspace(0, 1, 101)
        liney = np.sin(2 * np.pi * linex)
        subplot.plot(linex, liney, color='green', linestyle='--')

        linex = np.linspace(0,1,101)
        # like probabilty density function
        liney = f(linex)
        label = "E(RMS)=%.2f" % rmse(train_set, f)
        subplot.plot(linex, liney, color='red', label=label)
        subplot.legend(loc=1)

    print("Table of the coeffcients")
    print(df_ws.transpose())
    fig.show()

    df = DataFrame()
    for m in range(0, 10):
        f, ws = resolve(train_set, m)
        train_error = rmse(train_set, f)
        test_error = rmse(test_set, f)
        df = df.append(
                Series([train_error, test_error], index=['Training set', 'Test set']),
                ignore_index=True)

    df.plot(title='RMS Error', style=['-', '--'], grid=True, ylim=(0, 0.9))
    plt.show()

if __name__ == '__main__':
    main()
