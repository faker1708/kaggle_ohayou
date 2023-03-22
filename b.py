
# https://blog.csdn.net/splnn/article/details/121469289

import numpy as np
import pandas as pd
import matplotlib as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


dftrain_raw = pd.read_csv('titanic/train.csv')
dftest_raw = pd.read_csv('titanic/test.csv')
PassengerId = dftest_raw['PassengerId']
# print(dftrain_raw.head(10))


def preprocessing(dfdata):

    dfresult= pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)


x_temp = preprocessing(dftrain_raw)
# print(x_temp)

x_train = preprocessing(dftrain_raw).values
y_train = dftrain_raw[['Survived']].values

x_test = preprocessing(dftest_raw).values
# print("x_train.shape =", x_train.shape )
# print("y_train.shape =", y_train.shape )

# print("x_test.shape =", x_test.shape)

dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float()),
                     shuffle = True, batch_size = 8)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.FloatTensor):
        return self.net(x)
net = Net()
# print(net)


from torchkeras import summary
summary(net,input_shape=(15,))


optim = torch.optim.Adam(Net.parameters(net), lr=0.001)
Loss = nn.MSELoss()


for epoch in range(100):
    loss = None
    for batch_x, batch_y in dl_train:
        y_predict = net(batch_x)
        loss = Loss(y_predict, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    if (epoch % 10 == 0):
        print("epoch: %d => loss: %f" % (epoch, loss.item()))
predict = net(torch.tensor(x_test, dtype=torch.float))
predict = predict.detach().numpy()
predict = predict.reshape(418)
print(predict.shape)
print(PassengerId.shape)


predict = np.round(predict)
predict = predict.astype(int)


submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predict})
submission.to_csv("titanic-submission.csv", index=False)
