"""
Created by Haochuan Lu on 6/5/17.
"""
import FCNet
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

def down_sample(df):
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


def normalize(data):
    divide = np.max(data,axis=0) - np.min(data,axis=0)
    divide[divide==0] = 1
    minus = np.min(data,axis=0)
    result = data - minus
    result /= divide
    #print(result)
    return result

def feature_select(X,y):
    mask = SelectFromModel(GradientBoostingClassifier()).fit(X,y).get_support()
    return mask


data_train = pd.read_csv('data/train.csv',index_col='ID')
XX = data_train.drop(['TARGET'],axis=1)
X_f = np.array(XX)
y_f = np.array(data_train['TARGET'])
data_feature = feature_select(X_f,y_f)

XX = XX.iloc[:,data_feature]
XX['TARGET'] = data_train['TARGET']
data_train = XX
#print(data_train.columns)
input_dim = data_train.shape[1] -1
print(input_dim)
fc_net = FCNet.FCNet(input_dim,[70,80,90,100,110,120,130,140,150,160,170,180,190,200,190,180,170,160,150,140,130,120,110,100,90,80,70,60,50,40,30,20],2)
lr = 3e-5
for i in range(30):
    data = down_sample(data_train)
    X = data.drop(['TARGET'],axis=1)
    yy = data['TARGET']
    y = pd.get_dummies(yy)
    X_batch = np.array(X,dtype='float')
    y_batch = np.array(y,dtype='float')
    X_batch = normalize(X_batch)
    fc_net.train(X_batch, y_batch,learning_rate=lr,batch_size=800,trained_model_path='model/model_3.ckpt')
    lr *=0.95

data_test = pd.read_csv('data/test.csv',index_col='ID')

data_test = data_test.iloc[:,data_feature]


X_test = data_test
X_array = np.array(X_test)
X_array = normalize(X_array)
y_test = fc_net.predict(X_array,model_path='model/model_3.ckpt')
X_test['TARGET'] = y_test
result = X_test[['TARGET']]
result.to_csv('data/predictions.csv')
