import sys
import numpy as np
import pandas as pd
from optparse import OptionParser
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import load_model

def train(trainPath='../data2/stock_train_data_20170910.csv',modelPath='model.h5'):
    train = pd.read_csv(trainPath)
    columnsTrain = ['feature'+str(i) for i in range(0,88)]
    columnsTrain.append('group')
    columnsTrain.append('weight')

    X = np.array(train.ix[:,columnsTrain])
    scaler = preprocessing.StandardScaler().fit_transform(X)

    Y = np.array(train.ix[:,['label']])

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=1)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1]-1, init='uniform', activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train[:,0:X_train.shape[1]-1],Y_train,
              batch_size=128,
              epochs=40,
              sample_weight=X_train[:,X_train.shape[1]-1])
    model.save(modelPath)
    score = model.evaluate(X_test[:,0:X_test.shape[1]-1], Y_test,sample_weight=X_test[:,X_test.shape[1]-1],verbose=1)
    #score = model.evaluate(X_test[:,0:X_test.shape[1]-1], Y_test,sample_weight=None,verbose=1)
    print(score)


def getProba(testPath,modelPath):
    test = pd.read_csv(testPath)
    testID = test.ix[:,'id']
    columnsTest = list(test.columns)
    columnsTest.remove('id')
    test = np.array(test.ix[:,columnsTest])
    model = load_model(modelPath)
    proba = model.predict_proba(test,batch_size=128)
    proba = pd.Series(proba[:,0])
    result = pd.concat([testID,proba],axis=1)
    result.columns = ['id','proba']
    result.to_csv('data.csv',index=False)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-t','--task',dest='task',default='train')
    (options,args)=parser.parse_args(sys.argv)
    
    if options.task == 'train':
        train()
    elif options.task == 'test':
        getProba(testPath = '../data2/stock_test_data_20170910.csv',
                 modelPath = 'model.h5')
