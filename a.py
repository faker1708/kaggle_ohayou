import pandas as pd
import torch
import numpy as np

data_train = pd.read_csv('./titanic/train.csv')
data_test = pd.read_csv('./titanic/test.csv')

# print(data_train.describe())

a = data_train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)


a = data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(a)
# sns.catplot(x='Pclass',col='Embarked',y='Survived',hue='Sex',data=data_train,kind='point')

#删除'PassengerId','Ticket', 'Cabin'这三个特征
data_train = data_train.drop(['PassengerId','Ticket', 'Cabin'], axis=1)
data_test = data_test.drop(['PassengerId','Ticket', 'Cabin'], axis=1)

print(data_train.info())


print('train.csv缺失值比例：',data_train.isnull().sum()/data_train.shape[0],sep='\n')
print('test.csv缺失值比例：',data_test.isnull().sum()/data_test.shape[0],sep='\n')
