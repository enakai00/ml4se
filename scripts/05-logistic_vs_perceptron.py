# -*- coding: utf-8 -*-
#
# ロジスティック回帰とパーセプトロンの比較
#
# 2015/04/24 ver1.0
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import rand, multivariate_normal
pd.options.mode.chained_assignment = None

#------------#
# Parameters #
#------------#
Variances = [5,10,30,50] # 両クラス共通の分散（4種類の分散で計算を実施）


# データセット {x_n,y_n,type_n} を用意
def prepare_dataset(variance):
    n1 = 10
    n2 = 10
    mu1 = [7,7]
    mu2 = [-3,-3]
    cov1 = np.array([[variance,0],[0,variance]])
    cov2 = np.array([[variance,0],[0,variance]])

    df1 = DataFrame(multivariate_normal(mu1,cov1,n1),columns=['x','y'])
    df1['type'] = 1
    df2 = DataFrame(multivariate_normal(mu2,cov2,n2),columns=['x','y'])
    df2['type'] = 0
    df = pd.concat([df1,df2],ignore_index=True)
    df = df.reindex(np.random.permutation(df.index)).reset_index()
    return df[['x','y','type']]

# ロジスティック回帰
def run_logistic(tset, subplot):
    w = np.array([[0],[0.1],[0.1]])
    phi = tset[['x','y']]
    phi['bias'] = 1
    phi = phi.as_matrix(columns=['bias','x','y'])
    t = tset[['type']]
    t = t.as_matrix()

    # 最大30回のIterationを実施
    for i in range(30):
        # IRLS法によるパラメータの修正
        y = np.array([])
        for line in phi:
            a = np.dot(line, w)
            y = np.append(y, [1.0/(1.0+np.exp(-a))])
        r = np.diag(y*(1-y)) 
        y = y[np.newaxis,:].T
        tmp1 = np.linalg.inv(np.dot(np.dot(phi.T, r),phi))
        tmp2 = np.dot(phi.T, (y-t))
        w_new = w - np.dot(tmp1, tmp2)
        # パラメータの変化が 0.1% 未満になったら終了
        if np.dot((w_new-w).T, (w_new-w)) < 0.001 * np.dot(w.T, w):
            w = w_new
            break
        w = w_new

    # 分類誤差の計算
    w0, w1, w2 = w[0], w[1], w[2]
    err = 0
    for index, point in tset.iterrows():
        x, y, type = point.x, point.y, point.type
        type = type * 2 - 1
        if type * (w0 + w1*x + w2*y) < 0:
            err += 1
    err_rate = err * 100 / len(tset)

    # 結果を表示
    xmin, xmax = tset.x.min()-5, tset.x.max()+10
    linex = np.arange(xmin-5, xmax+5)
    liney = - linex * w1 / w2 - w0 / w2
    label = "ERR %.2f%%" % err_rate
    subplot.plot(linex,liney ,label=label, color='blue')
    subplot.legend(loc=1)

# パーセプトロン
def run_perceptron(tset, subplot):
    w0 = w1 = w2 = 0.0
    bias = 0.5 * (tset.x.mean() + tset.y.mean())

    # Iterationを30回実施
    for i in range(30):
        # 確率的勾配降下法によるパラメータの修正
        for index, point in tset.iterrows():
            x, y, type = point.x, point.y, point.type
            type = type*2-1
            if type * (w0*bias + w1*x + w2*y) <= 0:
                w0 += type * 1
                w1 += type * x
                w2 += type * y
    # 分類誤差の計算
    err = 0
    for index, point in tset.iterrows():
        x, y, type = point.x, point.y, point.type
        type = type*2-1
        if type * (w0*bias + w1*x + w2*y) <= 0:
            err += 1
    err_rate = err * 100 / len(tset)

    # 結果を表示
    xmin, xmax = tset.x.min()-5, tset.x.max()+10
    linex = np.arange(xmin-5, xmax+5)
    liney = - linex * w1 / w2 - bias * w0 / w2
    label = "ERR %.2f%%" % err_rate
    subplot.plot(linex, liney, label=label, color='red', linestyle='--')
    subplot.legend(loc=1)

# データを準備してロジスティック回帰とパーセプトロンを実行
def run_simulation(variance, subplot):
    tset = prepare_dataset(variance)
    tset1 = tset[tset['type']==1]
    tset2 = tset[tset['type']==0]
    ymin, ymax = tset.y.min()-5, tset.y.max()+10
    xmin, xmax = tset.x.min()-5, tset.x.max()+10
    subplot.set_ylim([ymin-1, ymax+1])
    subplot.set_xlim([xmin-1, xmax+1])
    subplot.scatter(tset1.x, tset1.y, marker='o')
    subplot.scatter(tset2.x, tset2.y, marker='x')

    run_logistic(tset, subplot)
    run_perceptron(tset, subplot)

# Main
if __name__ == '__main__':
    fig = plt.figure()
    plt.suptitle('Blue: Logistic Regression, Red: Perceptron')
    for c, variance in enumerate(Variances):
        subplot = fig.add_subplot(2,2,c+1)
        run_simulation(variance, subplot)
    fig.show()
