#import libraries
from sklearn.datasets import load_wine, load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd


#%%
# Generate dataset
def generate_data(dataset):
  if dataset=='Wine':
    data = load_wine()
  elif dataset == 'Iris':
    data = load_iris()
  df = pd.DataFrame(data.data, columns=data.feature_names , index=None)
  df['Type'] = data.target
  X = data.data
  y = data.target
  X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=1, test_size=0.2)
  return X, y, X_train, X_test, y_train, y_test,df,data.target_names

#%%

X, y, X_train, X_test, y_train, y_test, df , classes= generate_data('Wine')

#%%
#classifiers
dt_clf = DecisionTreeClassifier()
svc_clf = SVC()
kn_clf = KNeighborsClassifier()
rf_clf = RandomForestClassifier()


classifiers = [dt_clf, svc_clf, kn_clf, rf_clf]

for clf in classifiers:
  print(f'----{clf}---------')
  print("fitting - training...")
  clf.fit(X_train,y_train)

  print("training on whole dataset...")
  clf.fit(X, y)

  print("predicting...")
  y_pred = clf.predict(X_test)

  # wypisujemy wartości dla pierwsyzch 10 predykcji

  print("true values ", y[:10])
  print("predicted   ", y_pred[:10])

  print("scoring...")

  clf_score = clf.score(X_train,y_train)
  print("Train score = ", clf_score)
  clf_score = clf.score(X_test,y_test)
  print("Test score = ", clf_score)

  clf_score = clf.score(X,y)
  print("whole set score = ", clf_score)
  
