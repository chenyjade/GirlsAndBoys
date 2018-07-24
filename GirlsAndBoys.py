from __future__ import division

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
import gc
from scipy.stats import mode
import math
import re


from datetime import datetime

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')


df_train.drop(['In_30','In_31','In_42','In_43','In_44','In_45','In_64','In_65','In_66','In_67','In_68','In_69','In_70','In_71','In_72','In_73','In_74','In_75','In_249'],axis=1, inplace=True)
df_test.drop(['In_30','In_31','In_42','In_43','In_44','In_45','In_64','In_65','In_66','In_67','In_68','In_69','In_70','In_71','In_72','In_73','In_74','In_75','In_249'],axis=1, inplace=True)

df_train['time']=(pd.to_datetime(df_train['In_209'].map(lambda x: x[11:19]), format='%H.%M.%S') - datetime.strptime('00:00:00','%H:%M:%S')).map(lambda x:x.seconds)
df_test['time']=(pd.to_datetime(df_test['In_209'].map(lambda x: x[11:19]), format='%H.%M.%S') - datetime.strptime('00:00:00','%H:%M:%S')).map(lambda x:x.seconds)

df_train['hour'] = pd.to_datetime(df_train['In_209'].map(lambda x: x[11:19]), format='%H.%M.%S').map(lambda x:x.hour)
df_test['hour'] = pd.to_datetime(df_test['In_209'].map(lambda x: x[11:19]), format='%H.%M.%S').map(lambda x:x.hour)

df_train['minute'] = pd.to_datetime(df_train['In_209'].map(lambda x: x[11:19]), format='%H.%M.%S').map(lambda x:x.minute)
df_test['minute'] = pd.to_datetime(df_test['In_209'].map(lambda x: x[11:19]), format='%H.%M.%S').map(lambda x:x.minute)

df_train['day']=pd.to_datetime(df_train['In_209'].map(lambda x: x[0:10]), format='%Y-%m-%d').map(lambda x:x.day)
df_test['day']=pd.to_datetime(df_test['In_209'].map(lambda x: x[0:10]), format='%Y-%m-%d').map(lambda x:x.day)

df_train['month']=pd.to_datetime(df_train['In_209'].map(lambda x: x[0:10]), format='%Y-%m-%d').map(lambda x:x.month)
df_test['month']=pd.to_datetime(df_test['In_209'].map(lambda x: x[0:10]), format='%Y-%m-%d').map(lambda x:x.month)


df_train['In_209']=(pd.to_datetime(df_train['In_209'].map(lambda x: x[0:10]), format='%Y-%m-%d') - pd.datetime(2017,1,1)).map(lambda x:x.days)
df_test['In_209']=( pd.to_datetime(df_test['In_209'].map(lambda x: x[0:10]), format='%Y-%m-%d') - pd.datetime(2017,1,1)).map(lambda x:x.days)

df_train['week'] = df_train['In_209'] % 7
df_test['week'] = df_test['In_209'] % 7

def day_time(hour):
    if (hour < 12):
        return 1
    if (hour < 18):
        return 2
    else:
        return 3

df_train['day_time'] = df_train['hour'].map(day_time)
df_test['day_time'] = df_test['hour'].map(day_time)


def str2int(strs):
    ans = 0
    tmp = 1
    for i in list(strs):
        ans += (ord(i)-64)*tmp
        tmp *= 26
    return  ans

from sklearn.preprocessing import LabelEncoder
onehot = LabelEncoder()
for i in ['In_135','In_138','In_141','In_144','In_147','In_150','In_153','In_156','In_159','In_162','In_165','In_168','In_171','In_174','In_177','In_180','In_183','In_186','In_189','In_192','In_195','In_198','In_201','In_204']:
    df_train[i]=df_train[i].fillna('UNKNOWN0')
    df_test[i]=df_test[i].fillna('UNKNOWN0')

    df_train[i+'str'] =  df_train[i].astype('str').map(lambda x: re.findall(r'[0-9]+|[A-Z]+',x.upper())[0])
    df_train[i+'num'] =  df_train[i].astype('str').map(lambda x: re.findall(r'[0-9]+|[A-Z]+',x.upper())[1])

    df_train[i+'strlen'] = df_train[i+'str'].astype('str').map(lambda x: len(x))
    df_train[i+'numlen'] = df_train[i+'num'].astype('str').map(lambda x: len(x))

    df_test[i+'str'] =  df_test[i].astype('str').map(lambda x: re.findall(r'[0-9]+|[A-Z]+',x.upper())[0])
    df_test[i+'num'] =  df_test[i].astype('str').map(lambda x: re.findall(r'[0-9]+|[A-Z]+',x.upper())[1])

    df_test[i+'strlen'] = df_test[i+'str'].astype('str').map(lambda x: len(x))
    df_test[i+'numlen'] = df_test[i+'num'].astype('str').map(lambda x: len(x))

   
    df_train[i+'4'] = df_train[i+'num'].astype('str').map(lambda x: 0 if len(x) < 2 else int(x[-2]))
    df_train[i+'5'] = df_train[i+'num'].astype('str').map(lambda x: int(x[-1]))

    df_test[i+'4'] = df_test[i+'num'].astype('str').map(lambda x: 0 if len(x) < 2 else int(x[-2]))
    df_test[i+'5'] = df_test[i+'num'].astype('str').map(lambda x: int(x[-1]))

    df_train[i+'str'] = df_train[i+'str'].astype('str').map(str2int)
    df_test[i+'str'] = df_test[i+'str'].astype('str').map(str2int)

    df_train[i+'num'] = df_train[i+'num'].astype('str').map(lambda x: int(x))
    df_test[i+'num'] = df_test[i+'num'].astype('str').map(lambda x: int(x))

    df_train[i]=df_train[i].astype('str').map(str2int)
    df_test[i]=df_test[i].astype('str').map(str2int)

df_train['In_210_1'] = df_train['In_210'].astype('str').map(lambda x: x[:2]).map(str2int)
df_test['In_210_1'] = df_test['In_210'].astype('str').map(lambda x: x[:2]).map(str2int)

df_train['In_210_2'] = df_train['In_210'].astype('str').map(lambda x: int(x[2:6]))
df_test['In_210_2'] = df_test['In_210'].astype('str').map(lambda x: int(x[2:6]))

df_train['In_210_3'] = df_train['In_210'].astype('str').map(lambda x: x[6]).map(str2int)
df_test['In_210_3'] = df_test['In_210'].astype('str').map(lambda x: x[6]).map(str2int)

df_train['In_210_4'] = df_train['In_210'].astype('str').map(lambda x: int(x[7]) if x[7] != 'A' else 11)
df_test['In_210_4'] = df_test['In_210'].astype('str').map(lambda x: int(x[7]) if x[7] != 'A' else 11)

df_train['In_210_5'] = df_train['In_210'].astype('str').map(lambda x: x[-2:]).map(str2int)
df_test['In_210_5'] = df_test['In_210'].astype('str').map(lambda x: x[-2:]).map(str2int)


df_train['In_210'] = df_train['In_210'].astype('str').map(str2int)
df_test['In_210'] = df_test['In_210'].astype('str').map(str2int)

df_train['In_247_1'] =  df_train['In_247'].astype('str').map(lambda x: x.split('-')[0] if x != 'UNKNOWN' else x).map(str2int)
df_train['In_247_2'] =  df_train['In_247'].astype('str').map(lambda x: x.split('-')[1] if x != 'UNKNOWN' else x).map(str2int)
df_train['In_247_3'] =  df_train['In_247'].astype('str').map(lambda x: int(x.split('-')[2]) if x != 'UNKNOWN' else -1)
df_train['In_247_4'] =  df_train['In_247'].astype('str').map(lambda x: x.split('-')[0][:2]).map(str2int)
df_train['In_247_5'] =  df_train['In_247'].astype('str').map(lambda x: int(x.split('-')[0][2]) if x != 'UNKNOWN' else -1)

df_test['In_247_1'] =  df_test['In_247'].astype('str').map(lambda x: x.split('-')[0] if x != 'UNKNOWN' else x).map(str2int)
df_test['In_247_2'] =  df_test['In_247'].astype('str').map(lambda x: x.split('-')[1] if x != 'UNKNOWN' else x).map(str2int)
df_test['In_247_3'] =  df_test['In_247'].astype('str').map(lambda x: int(x.split('-')[2]) if x != 'UNKNOWN' else -1)
df_test['In_247_4'] =  df_test['In_247'].astype('str').map(lambda x: x.split('-')[0][:2]).map(str2int)
df_test['In_247_5'] =  df_test['In_247'].astype('str').map(lambda x: int(x.split('-')[0][2]) if x != 'UNKNOWN' else -1)

df_train['In_247'] = df_train['In_247'].astype('str').map(str2int)
df_test['In_247'] = df_test['In_247'].astype('str').map(str2int)

df_train = df_train.fillna(np.nan)
df_test = df_test.fillna(np.nan)

df_test['nan'] = df_test.isnull().sum(axis=1)
df_train['nan'] = df_train.isnull().sum(axis=1)

remove = ['In_85', 'In_86', 'In_110', 'In_114', 'In_119', 'In_158', 'In_161', 'In_164', 'In_167', 'In_170', 'In_173', 'In_176', 'In_185', 'In_188', 'In_191', 'In_194', 'In_197', 'In_200', 'In_203', 'In_206', 'In_215', 'In_235', 'In_240', 'In_241', 'In_242', 'In_243', 'In_244', 'In_245', 'In_246']
df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)



remove = []
for c in df_train.columns:
    if df_train[c].isnull().sum() > 10000:
        remove.append(c)
df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)


from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

cp_train = df_train.copy().drop(['label','Id'], axis=1)
cp_test = df_test.copy().drop(['Id'], axis=1)


df_test['n0'] = (cp_test==0).sum(axis=1)
df_train['n0'] = (cp_train==0).sum(axis=1)


len_train = len(cp_train)
len_test = len(cp_test)
cp_train = cp_train.append(cp_test)
for col in cp_train.columns:
    cp_train[col] = cp_train[col].fillna(cp_train[col].mean())


from sklearn.preprocessing import MinMaxScaler
# PCA
X_normalized = normalize(cp_train, axis=1)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_normalized)

df_train['PCA1'] = X_pca[:,0][:len_train]
df_train['PCA2'] = X_pca[:,1][:len_train]
df_train['PCA3'] = X_pca[:,2][:len_train]

Y_train = df_train['label'].map(lambda x: 0 if x == -1 else 1).values


df_test['PCA1'] = X_pca[:,0][len_train:]
df_test['PCA2'] = X_pca[:,1][len_train:]
df_test['PCA3'] = X_pca[:,2][len_train:]

cp_test = cp_train[len_train:]
cp_train = cp_train[:len_train]


X_train = df_train.drop(['Id','label'], axis=1).values
id_test = df_test['Id']
X_test = df_test.drop(['Id'], axis=1).values


len_train = len(X_train)
len_test  = len(X_test)

ratio = float(np.sum(Y_train==0)) / (np.sum(Y_train==1))
print ("ratio: ", ratio)

name = list(df_test.columns)
name.remove('Id')
print (len(name))
categorical = ['In_135','In_138','In_141','In_144','In_147','In_150','In_153','In_156','In_159','In_162','In_165','In_163','In_168','In_171','In_174','In_177','In_180','In_183','In_186','In_189','In_192','In_195','In_198','In_199','In_201','In_204','In_210','In_247']
gc.collect()

def inv_sigmoid(x):
    return math.log(x/(1-x))

from sklearn.model_selection import KFold, StratifiedKFold

clf = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',num_leaves=330, n_estimators=6000, learning_rate=0.002, min_child_weight=4, subsample=0.82, colsample_bytree=0.65,random_state=233, n_jobs=30, scale_pos_weight=1,max_bin=300, min_child_samples=20, reg_lambda=1) # 0.838
kf = StratifiedKFold(n_splits=20, shuffle=True, random_state=233)


predict = np.zeros((len_test,))
oof_predict = np.zeros((len_train,))
scores = []
count = 0
sample_submission = pd.DataFrame.from_dict({'Id': id_test})
sample_submission['label'] = predict
sample_submission['tmp'] = predict

for train_index, test_index in kf.split(X_train, Y_train):

    y_train,y_test = Y_train[train_index], Y_train[test_index]
    kfold_X_train = X_train[train_index]
    kfold_X_valid = X_train[test_index]
    bst = clf.fit(kfold_X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=[(kfold_X_valid, y_test)], verbose=200)

    sample_submission['tmp'] = bst.predict_proba(X_test, num_iteration=bst.best_iteration_)[:,1]
    sample_submission['label'] += sample_submission['tmp'] / 20
    oof_predict[test_index] = bst.predict_proba(kfold_X_valid, num_iteration=bst.best_iteration_)[:,1] #model.predict(kfold_X_valid, batch_size=512)
    cv_score = roc_auc_score(y_test, oof_predict[test_index])

    scores.append(cv_score)

    print('score: ',cv_score)

print('Total CV score is {}'.format(np.mean(scores)))
oof = pd.DataFrame.from_dict({'Id': df_train['Id']})

print (sample_submission['label'].describe())
sample_submission.drop(['tmp'], axis=1, inplace=True)
sample_submission.to_csv('k_f9k_folds.csv', index=False)

oof['label'] = oof_predict
oof.to_csv('k_f9k_oof.csv', index=False)


print('Overall AUC:', roc_auc_score(Y_train,oof_predict))
