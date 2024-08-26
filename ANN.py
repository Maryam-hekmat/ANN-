"""
building the ANN
"""

"""
step 1
"""
#preprocessing

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
dataset=pd.read_csv("Churn_Modelling.csv")
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

print(x)
print(y)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                        remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                                    random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""
step 2
"""
#Building the ANN


# Initializing the ANN
import tensorflow as tf

A_nn = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
A_nn.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
A_nn.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
A_nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""
step 3
"""
#Training the ANN

# Compiling the ANN
A_nn.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# Training the ANN on the Training set
A_nn.fit(x_train, y_train, batch_size = 32, epochs = 100)

"""
step 4
"""

# Making the predictions and evaluating the model

# Predicting the result of a single observation
# """
# a person
# France
# credit card=600
# gender=male
# age=40
# tenure=3years
# balance=60000$
# number of products=2
# having credit card or not=yes
# est.salary=50000$
# predicts whether he has chance to stay in the bank or not?

# """
print(A_nn.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000,
                                  2, 1, 1, 50000]])) > 0.5)








# Predicting the Test set results
y_pred = A_nn.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),
                      y_test.reshape(len(y_test),1)),1))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


