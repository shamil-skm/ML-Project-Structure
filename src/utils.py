import sys
import os
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report ={}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train)  #Train Model

            y_train_pred = model.predict(X_train)  
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

