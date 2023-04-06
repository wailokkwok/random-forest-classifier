import itertools

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

df = pd.read_csv("creditcard.csv")
pd.options.display.max_columns = None

X = df.drop("Class", axis = 1)
Y = df["Class"]

# checked correlation
#corr = X.corr()
#corr = corr.abs().unstack()
#corr = corr.sort_values(ascending = False)
#corr = corr[corr< 1]


X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# random select

zero_x = []
zero_y = []
one_x = []
one_y = []
for index, _ in enumerate(y_train):
    if _ == 0:
        zero_y.append(0)
        zero_x.append(X_train[index])

    else:
        one_y.append(1)
        one_x.append(X_train[index])

random_zero = np.random.randint(190477, size=(400))
random_one = np.random.randint(343, size=(400))

balance_x = []
balance_y = []
for z in random_zero:
    balance_x.append(zero_x[z])
    balance_y.append(zero_y[z])

for a in random_one:
    balance_x.append(one_x[a])
    balance_y.append(one_y[a])

# 400 0 and 400 1

train_x, train_y = shuffle(np.array(balance_x), np.array(balance_y))

#NN
model = Input(shape = (30))
dense = Dense(32, activation = "relu")(model)
output = Dense(1, activation = "sigmoid")(dense)
history = Model(model, output)
history.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

model = history.fit(train_x, train_y, validation_split=0.33, epochs=10, batch_size=10, verbose=0)


clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(train_x, train_y)
clf.score(train_x, train_y)
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))