import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_model,save_object

@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()
    

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting the training set into train and test set")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Random Forest Regressor': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boost Regressor': GradientBoostingRegressor(),
                'Linear Regressor': LinearRegression(),
                'XG Boost Regressor': XGBRegressor(),
                'Cat Boost Regressor': CatBoostRegressor(verbose=False),
                'AdaBoost Regressor': AdaBoostRegressor(),
            }

            model_report = evaluate_model(X_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test, models=models)

            best_model_score =  max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # if best_model_score<0.6:
            #     raise CustomException('No  best Model Found')

            logging.info("Best model found for both training and testing")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_value =  r2_score(y_test,predicted)
            print(best_model_name)
            return r2_value

        except Exception as e:
            raise CustomException(e,sys)