import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#projects folder path

BASE_DIR=Path(__file__).resolve().parent.parent

#dataset load karo

data=pd.read_csv(BASE_DIR/"train.csv")

#input colums select karo

X=data[[
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare"]]

#target colum

y=data["Survived"]

#gender ko number mein convert karo

X=X.copy()

X["Sex"]=X["Sex"].map({
    "male":0,
    "female":1
})

#missing values fill karo

X["Age"]=X["Age"].fillna(X["Age"].median())
X["Fare"]=X["Fare"].fillna(X["Fare"].median())

#teaining aur testing data

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,
    random_state=42
)

#random forest model

model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

#model train karo

model.fit(X_train,y_train)

#accuracy check karo

y_pred=model.predict(X_test)

print("accuracy",accuracy_score(y_test,y_pred)*100)

#model save

joblib.dump(
    model,BASE_DIR/"prediction"/"titanic_model.pkl"
)

print("model saved succesfully")