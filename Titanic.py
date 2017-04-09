import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
dataset_answer = pd.read_csv('gender_submission.csv')

dataset_train["Embarked"] = dataset_train["Embarked"].fillna("S")
dataset_test["Embarked"] = dataset_test["Embarked"].fillna("S")

train = dataset_train.iloc[:,[2,4,5,6,7,9,11]].values
y_train = dataset_train.iloc[:, 1].values
test = dataset_test.iloc[:,[1,3,4,5,6,8,10]].values
y_test = dataset_answer.iloc[:,[1]].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0 ) #axis = 0 for taking mean across col and axis=1 to take mean across rows
imputer = imputer.fit(train[:, [2]]);
train[:,[2]] = imputer.transform(train[:,[2]])

imputer_1 = Imputer(missing_values = "NaN", strategy = "mean", axis = 0 ) #axis = 0 for taking mean across col and axis=1 to take mean across rows
imputer_1 = imputer_1.fit(test[:, [2]]);
test[:,[2]] = imputer_1.transform(test[:,[2]])

imputer_2 = Imputer(missing_values = "NaN", strategy = "mean", axis = 0 )
imputer_2 = imputer_2.fit(test[:, [5]]);
test[:,[5]] = imputer_2.transform(test[:, [5]])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
train[:,1] = labelencoder_X_1.fit_transform(train[:,1])
labelencoder_X_2 = LabelEncoder()
train[:,6] = labelencoder_X_2.fit_transform(train[:,6])
onehotencoder = OneHotEncoder(categorical_features = [6])
train = onehotencoder.fit_transform(train).toarray()

'''
import math

for i in range(0,418):
    for j in range(0,7):
        if math.isnan(test[i,j]):
            print str(i)+', '+str(j)
'''

labelencoder_X1_1 = LabelEncoder()
test[:,1] = labelencoder_X1_1.fit_transform(test[:,1])
labelencoder_X1_2 = LabelEncoder()
test[:,6] = labelencoder_X1_2.fit_transform(test[:,6])
onehotencoders = OneHotEncoder(categorical_features = [6])
test= onehotencoders.fit_transform (test).toarray()

train = train[:, 1:]
test = test[:, 1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)



import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising ANN
classifier = Sequential()

# Adding input and first hidden layer
classifier.add(Dense(output_dim = , init = 'uniform', activation = 'relu', input_dim = 8))

# Addind the second hidden layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to training set
classifier.fit(train, y_train, batch_size = 10, nb_epoch = 100)

# PART 3 Evaluation of model

# Predicting the Test set results
y_pred = classifier.predict(test)
y_pred = (y_pred > 0.5)

'''
# Applying the K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = train, y = y_train, cv = 20)
print accuracies.mean()
accuracies.std()


# Applying grid search to find the best model and parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10], 'kernel': ['linear'],
               'C': [0.5, 0.7, 1], 'kernel': ['rbf'], 'gamma': [0.1,0.3,0.5]}
              ]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
accuracy = ((cm[0,1] + cm[1,0])/float(cm[0,0] + cm[1,1]))
print str((1 - accuracy)*100)
'''

predictions = np.zeros((418,2))
predictions = np.matrix(predictions)

for i in range(892, 1310):
    predictions[i - 892, 0] = i
    predictions[i - 892, 1] = y_pred[i - 892]

predictions = np.asarray(predictions)
np.savetxt('Predictions.csv',predictions, header = 'PassengerId,Survived', delimiter = ',', fmt='%.0f')