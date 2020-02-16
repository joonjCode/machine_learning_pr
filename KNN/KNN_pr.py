import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')
print(data.head(5))

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
persons = le.fit_transform(list(data['persons']))
door = le.fit_transform(list(data['door']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_text, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
