"""
Created by Haochuan Lu on 6/5/17.
"""
import FCNet
import pandas as pd

data = pd.read_csv('data/train.csv',index_col='ID')
X = data.drop(['TARGET'],axis=1)
yy = data['TARGET']
y = pd.get_dummies(yy)
fc_net = FCNet.FCNet(369,[100,50],2)
X_batch = X[:50]
y_batch = y[:50]
fc_net.train(X_batch, y_batch,batch_size=20)

