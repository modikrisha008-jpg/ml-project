from types import prepare_class

from django.db.models.fields import return_None
from django.shortcuts import render
from pathlib import Path
import pandas as pd
import joblib
from sklearn.utils.graph import single_source_shortest_path_length

from titanic_project.settings import BASE_DIR

BASE_DIR = Path(__file__).resolve().parent.parent

model=joblib.load(
    BASE_DIR/"prediction"/"titanic_model.pkl"
)

# Create your views here.
def home(request):
    result=None
    if request.method=="POST":
        pclass=int(request.POST.get("pclass"))
        sex=request.POST.get("sex")
        age=int(request.POST.get("age"))
        sibsp=int(request.POST.get("sibsp"))
        parch=int(request.POST.get("parch"))
        fare=int(request.POST.get("fare"))
#convert gender into number
        sex_value=1 if sex == "female" else 0
#prepare input in the same formate as training data

        passenger=pd.DataFrame([{
         "Pclass":pclass,
          "Sex":sex_value,
          "Age":age,
          "SibSp":sibsp,
          "Parch":parch,
          "Fare":fare
}])
#predict survival

        prediction=model.predict(passenger)[0]

        if prediction==1:
            result="the passenger is predicted to survive"
        else:
            result="the passenger is predicted not to survive"

    return render(request,"prediction/home.html",{"result":result})