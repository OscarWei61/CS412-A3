import random
import math
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Do not import any other libraries

def get_splits(n, k, seed):
  splits = None
  # Implement your code to construct the splits here. Do NOT change the return type

  index = list(range(n))
  
  random.seed(seed)
  random.shuffle(index)
  
  each_fold_size = n // k

  splits = []
  start_index = 0

  for i in range(k):

    if (k - 1 - i) < ( n % k): 
      end_point = start_index + each_fold_size + 1
    else:
      end_point = start_index + each_fold_size + 0
    
    splits.append(index[start_index:end_point])
    start_index = end_point

  return splits

def my_cross_val(method, X, y, splits):
  errors = []
  # Implement your code to construct the list of errors here. Do NOT change the return type
  
  if method == 'LinearSVC':
    model = LinearSVC(max_iter=2000, random_state=412)
  elif method == 'SVC':
    model = SVC(gamma='scale', C=10, random_state=412)
  elif method == 'LogisticRegression':
    model = LogisticRegression(penalty='l2', solver='lbfgs', random_state=412, multi_class='multinomial')
  elif method == 'RandomForestClassifier':
    model = RandomForestClassifier(max_depth=20, n_estimators=500, random_state=412)
  elif method == 'XGBClassifier':
    model = XGBClassifier(max_depth=5, random_state=412)
  
  errors = []

  for ea_fold in splits:
    
    X_train = np.delete(X, ea_fold, axis=0)
    X_test = X[ea_fold]

    y_train = np.delete(y, ea_fold, axis=0)
    y_test = y[ea_fold]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    error_rate = np.mean(predictions != y_test)    
    errors.append(error_rate)

  return np.array(errors)