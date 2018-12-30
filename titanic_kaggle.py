# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:34:33 2018

@author: Adam
"""

# import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential

# load data
titanic=pd.read_csv('data/train.csv', sep=',', na_values='', index_col='PassengerId')

#Map male/female to 0/1
d={'male':0, 'female':1}
titanic['Sex']=titanic['Sex'].map(d)
print(titanic['Sex'].sample(5))


#Calculate the rate of survival for specific groups
def survivalrate(col, value):
### input: col : list of columns. value: list of values to check
    print('Survival rate for:' + col[0] + '=' + str(value[0]) + ':')
    b=(titanic[col[0]]==value[0])
    for r in range (1,len(col)):
        print('and ' + col[r] + '=' + str(value[r]) + ':')
        b=pd.concat([b,titanic[col[r]]==value[r]],axis=1,join_axes=[b.index])
    if len(col)>1:
        b2=b.all(axis=1)
    else:
        b2=b
    return titanic['Survived'].iloc[np.where(b2)].sum()/b2.sum()

print(survivalrate(['Sex'],[0]))
print(survivalrate(['Sex'],[1]))
print(survivalrate(['Pclass','Sex','Parch'],[1,1,1]))
print(survivalrate(['Pclass','Sex'],[1,0]))


plt.figure(1)
plt.cla()
titanic['Age'].plot.hist(bins=20)

plt.figure(2)
plt.cla()
titanic['Pclass'].plot.hist(bins=3)


plt.figure(3)
plt.cla()
m=(titanic['Sex'] == 0).sum()
f=(titanic['Sex'] == 1).sum()
ms=titanic['Survived'].iloc[np.where(titanic['Sex']==0)].sum()
fs=titanic['Survived'].iloc[np.where(titanic['Sex']==1)].sum()

plt.bar((0,1),(m,f),0.35)
plt.bar((0,1),(ms,fs),0.35)
plt.xticks((0,1),('male', 'female'))

plt.figure(4)
plt.cla()
first=(titanic['Pclass'] == 1).sum()
second=(titanic['Pclass'] == 2).sum()
third=(titanic['Pclass'] == 3).sum()

firsts=titanic['Survived'].iloc[np.where(titanic['Pclass']==1)].sum()
seconds=titanic['Survived'].iloc[np.where(titanic['Pclass']==2)].sum()
thirds=titanic['Survived'].iloc[np.where(titanic['Pclass']==3)].sum()

plt.bar((0,1,2),(first,second,third),0.35)
plt.bar((0,1,2),(firsts,seconds,thirds),0.35)
plt.xticks((0,1,2),('1st', '2nd', '3rd'))
