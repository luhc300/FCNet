"""
Created by Haochuan Lu on 6/5/17.
"""
import FCNet
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier


def down_sample(df,batch_size):
    '''
    number_default_1 = len(df[df.TARGET == 1])
    default_1 = np.array(df[df.TARGET == 1].index)
    default_0 = np.array(df[df.TARGET == 0].index)
    np.random.shuffle(default_0)
    random_default_0 = default_0[:int(number_default_1)]
    print(random_default_0)
    under_sample_default_0 = random_default_0
    under_sample_indices = np.concatenate([default_1, under_sample_default_0])
    # print(len(under_sample_indices))
    df_undersample = df.loc[under_sample_indices]
    return df_undersample
    '''
    number_default_1 = len(df[df.TARGET == 1])
    default_1 = np.array(df[df.TARGET == 1].index)
    default_0 = np.array(df[df.TARGET == 0].index)
    np.random.shuffle(default_0)
    np.random.shuffle(default_1)
    random_default_0 = default_0[:batch_size]
    random_default_1 = default_1[:batch_size]
    #print(random_default_0)
    under_sample_default_0 = random_default_0
    under_sample_default_1 = random_default_1
    under_sample_indices = np.concatenate([under_sample_default_1, under_sample_default_0])
    # print(len(under_sample_indices))
    np.random.shuffle(under_sample_indices)
    df_undersample = df.loc[under_sample_indices]
    return df_undersample

def normalize(data, d=None, m=None):
    if d is None:
        divide = np.max(data,axis=0) - np.min(data,axis=0)
        divide[divide==0] = 1
    else:
        divide = d
    if m is None:
        minus = np.min(data,axis=0)
    else:
        minus = m
    result = data - minus
    result /= divide
    #print(result)
    return result, divide, minus


def feature_select(X,y):
    mask = SelectFromModel(GradientBoostingClassifier(),'1.25*mean').fit(X,y).get_support()
    return mask

def data_wash(data):
    for index,col in data.iteritems():
        mean = np.mean(col)
        for i in range(0,len(col)):
            if np.abs(col.iloc[[i][0]]) > 900000 :
                col.iloc[[i][0]] = mean
    return data

def data_wash_2(data):
    for i in range(data.shape[1]):
        mean = np.mean(data[:,i])
        data[np.abs(data[:,i])>666666,i] = mean
    return data

def data_generate(train,test):
    index = train.shape[0]
    all = np.concatenate([train,test])
    all = data_wash_2(all)
    all = normalize(all)[0]
    train_gen = all[:index]
    test_gen = all[index:]
    return train_gen, test_gen


data_train = pd.read_csv('data/train.csv',index_col='ID')
'''
data_train = data_wash(data_train)
XX = data_train.drop(['TARGET'],axis=1)
X_f = np.array(XX)
y_f = np.array(data_train['TARGET'])
#data_feature = feature_select(X_f,y_f)

#XX = XX.iloc[:,data_feature]
XX['TARGET'] = data_train['TARGET']
data_train = XX
#print(data_train.columns)
input_dim = data_train.shape[1] -1
print(input_dim)
fc_net = FCNet.FCNet(input_dim,[500,400,300,200,100,50],2)
lr = 3e-5

data = data_train
X = data.drop(['TARGET'],axis=1)
yy = data['TARGET']
y = pd.get_dummies(yy)
X_batch = np.array(X,dtype='float')
y_batch = np.array(yy,dtype='float')
X_batch, divide, minus = normalize(X_batch)
print(X_batch)
#fc_net.train(X_batch, y_batch,learning_rate=lr,batch_size=800,trained_model_path='model/model_3.ckpt')
lr *=0.85

data_test = pd.read_csv('data/test.csv',index_col='ID')
data_test = data_wash(data_test)
#data_test = data_test.iloc[:,data_feature]


X_test = data_test
X_array = np.array(X_test)
X_array, divide2, minus2 = normalize(X_array, divide, minus)
y_test = fc_net.predict(X_array,model_path='model/model_3.ckpt')
X_test['TARGET'] = y_test
result = X_test[['TARGET']]
result.to_csv('data/predictions.csv')
'''
data_test = pd.read_csv('data/test.csv',index_col='ID')
XX = data_train.drop(['TARGET'],axis=1)
X_f = np.array(XX)
y_f = np.array(data_train['TARGET'])
X_t = np.array(data_test)
X_fg, X_tg = data_generate(X_f, X_t)
input_dim = X_fg.shape[1]
print(input_dim)
fc_net = FCNet.FCNet(input_dim,[500,50],2)
#fc_net.train(X_fg, y_f,learning_rate=1e-4,batch_size=800,trained_model_path='model/model_4.ckpt')
y_f_d = np.array(pd.get_dummies(y_f))
y_test = fc_net.predict(X_tg,model_path='model/model_3.ckpt')
print(y_test)
data_test['TARGET'] = y_test
result = data_test[['TARGET']]
result.to_csv('data/predictions.csv')