from src.utils.log_config import logger
from src.utils.exception import CustomException
import yaml
import os,sys,numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score



def read_yaml_file (file_path:str):
    try:
        with open (file_path , "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise CustomException(e,sys)


# def save_numpy_array_data(file_path: str, array:np.array):
#     try:
#         dir_path=os.path.dirname(file_path)
#         os.makedirs(dir_path,exist_ok=True)
#         with open(file_path,"wb") as file_obj:
#             np.save(file_obj,array)
#     except Exception as e:
#         raise CustomException(e,sys)

# def save_object(file_path:str,obj:object):
#     try:
#         logger.info('entered save object')
#         os.makedirs(os.path.dirname(file_path),exist_ok=True)
#         with open(file_path,"wb") as file_obj:
#             pickle.dump(obj,file_obj)
#         logger.info('completed save object')
#     except Exception as e:
#         raise CustomException(e,sys)

# def load_object(file_path:str):
#     try:
#         logger.info('entered load object')

#         if not os.path.exists(file_path):
#             raise Exception(f"The file: {file_path} doesnt exist")
#         with open(file_path,"rb") as file_obj:
#             print(file_obj)
#         logger.info('completed save object')
#     except Exception as e:
#         raise CustomException(e,sys)

# def load_numpy_array(file_path:str):
#     try:
#         with open(file_path , 'rb') as file_obj:
#             return np.load(file_obj)
#     except Exception as e:
#         raise CustomException(e,sys) from e

# def evaluate_models(X_train,y_train,X_test,y_test,models,param):
#     try:
#         report={}
#         for i in range (len(list(models))):
#             model=list(models.values())[i]
#             para=param[list(models.keys())[i]]

#             gs=GridSearchCV(model,para,cv=3)
#             gs.fit(X_train,y_train)
#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)

#             y_train_pred=model.predict(X_train)
#             y_test_pred=model.predict(X_test)
#             train_model_score=r2_score(y_train,y_train_pred)
#             test_model_score=r2_score(y_test,y_test_pred)
#             report[list(models.keys())[i]]=test_model_score

#             return report
    except Exception as e:
        raise CustomException(e,sys)