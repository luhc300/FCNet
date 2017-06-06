"""
Created by Haochuan Lu on 6/5/17.
"""
import FCNet
import pandas as pd
import numpy as np
def down_sample(df):
    number_default_1 = len(df[df.TARGET == 1])
    default_1 = np.array(df[df.TARGET == 1].index)
    default_0 = df[df.TARGET == 0].index
    random_default_0 = np.random.choice(default_0, int(number_default_1), replace=False)
    under_sample_default_0 = np.array(random_default_0)
    under_sample_indices = np.concatenate([default_1, under_sample_default_0])
    # print(len(under_sample_indices))
    df_undersample = df.loc[under_sample_indices]
    return df_undersample
def normalize(data):
    devide = np.max(data,axis=0) - np.min(data,axis=0)
    devide[devide==0] = 1
    minus = np.min(data,axis=0)
    result = data - minus
    result /= devide
    #print(result)
    return result
data = pd.read_csv('data/train.csv',index_col='ID')
data = down_sample(data)
X = data.drop(['TARGET'],axis=1)
yy = data['TARGET']
y = pd.get_dummies(yy)
fc_net = FCNet.FCNet(369,[500,450,400,350,300,250,200,150,100,50],2)
X_batch = np.array(X,dtype='float')
y_batch = np.array(y,dtype='float')
X_batch = normalize(X_batch)
fc_net.train(X_batch, y_batch,learning_rate=3e-5,batch_size=400)

