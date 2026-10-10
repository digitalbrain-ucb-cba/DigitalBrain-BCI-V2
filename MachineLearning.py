from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle as pkl
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split#, cross_val_score, GridSearchCV
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
csv1=""
csv2=""
nombremodelo="funka.pkl"

df0 = pd.read_csv(csv1, encoding="latin-1")
df1 = pd.read_csv(csv2, encoding="latin-1")
df = pd.concat([df0, df1])
df.drop(columns=["Unnamed: 0"], inplace=True)
X = df.iloc[:,0:16]
y = df.iloc[:,16]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

#rf_model = RandomForestClassifier(random_state=42)

#scores_rf = cross_val_score(rf_model, X_train, y_train, scoring='accuracy', cv=5)
#print("Cross-validation scores (RF):", scores_rf)
#print("Mean cross-validation score (RF):", scores_rf.mean())
#print("Standard deviation of cross-validation scores (RF):", scores_rf.std())

# pruebitaaas
#param_grid_rf = {
#    "n_estimators": [50, 100, 200],
#    "criterion": ["gini", "entropy"],
#    "max_depth": [2, 3, 4, 5, None],
#    "min_samples_split": [2, 4, 6, 8],
#    "min_samples_leaf": [1, 2, 3],
#}

#el coso ya probado con los mejores parametros
rf = RandomForestClassifier(
    criterion="entropy",
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=6,
    n_estimators=50,
    random_state=42
)

#grid_search_rf = GridSearchCV(
#    estimator=RandomForestClassifier(random_state=42),
#    param_grid=param_grid_rf,
#    cv=5,
#    scoring="accuracy",
#    n_jobs=-1
#)

#grid_search_rf.fit(X_train, y_train)

#print("Best parameters (RF):", grid_search_rf.best_params_)
#print("Best CV score (RF):", grid_search_rf.best_score_)

#Esto guarda el modelo
#rf = grid_search_rf.best_estimator_   #con los mejores parametros
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

with open(nombremodelo, "wb") as f:
    pkl.dump(rf, f)
