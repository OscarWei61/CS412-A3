import random
import math
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Do not import any other libraries

def my_train_test(method, X, y, pi, k):
    errors = []
    # Implement your code to construct the list of errors here. Do NOT change the return type
    
    if method == 'LinearSVC':  
        model = LinearSVC(max_iter=2000)
    elif method == 'SVC':
        model = SVC(gamma='scale', C=10)
    elif method == 'LogisticRegression':
        model = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial')
    elif method == 'RandomForestClassifier':
        model = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=500)
    elif method == 'XGBClassifier':
        model = XGBClassifier(max_depth=5)

    for k_index in range(k):
        
        index = list(range(len(X)))

        random.shuffle(index)

        train_data = index[:int( pi * len(X))]
        test_data = index[int( pi * len(X)):]

        x_train = X[train_data]
        y_train = y[train_data]

        x_test = X[test_data]
        y_test = y[test_data]

        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        errorrate = np.mean(pred != y_test)
        errors.append(errorrate)
        
    return np.array(errors)