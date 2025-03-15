import os, sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path= os.path.join("artifacts/Data_Transformation", 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        try:
            logging.info(f"Initiating Data Transformation")
            num_features=['age', 'workclass', 'education', 'education_number', 'marital_status',
       'occupation', 'relationship', 'race', 'sex', 'capital_gain',
       'capital_loss', 'hours_per_week', 'native_country']
            
            num_pipeline = Pipeline(
                steps = [
                ("imputer", SimpleImputer(strategy = 'median')),
                ("scaler", StandardScaler())

                
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_features)
            ])

            return preprocessor
            
        except Exception as e:
            raise CustomException(e)
    
    def remove_outliers_IQR(self, col, df):
        try:
            logging.info(f"Removing Outliers using IQR for {col}")
            df[col] = df[col].astype(float)  # Ensure float dtype to avoid dtype issues
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            logging.info(f"Outliers Removed for {col}")
            return df
        except Exception as e:
            logging.error(f"Error in remove_outliers_IQR: {str(e)}")
            raise CustomException(e)

    


    def initiateDataTransformation(self, train_path, test_path):
        try:
            logging.info("Initiating Data Transformation")

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            num_features = ['age', 'workclass', 'education', 'education_number', 'marital_status',
                            'occupation', 'relationship', 'race', 'sex', 'capital_gain',
                            'capital_loss', 'hours_per_week', 'native_country']

            # Apply outlier removal correctly
            for col in num_features:
                if col in train_data.columns:
                    train_data = self.remove_outliers_IQR(col, train_data)
                if col in test_data.columns:
                    test_data = self.remove_outliers_IQR(col, test_data)

            preprocess_obj = self.get_data_transformation_obj()
            target_col = 'income'
            drop_cols = [target_col]

            logging.info("Splitting train data into dependent and independent features")
            input_feature_train_data = train_data.drop(columns=drop_cols)
            target_feature_train_data = train_data[target_col]

            logging.info("Splitting test data into dependent and independent features")
            input_feature_test_data = test_data.drop(columns=drop_cols)
            target_feature_test_data = test_data[target_col]

            # Apply transformation on train and test data
            input_train_arr = preprocess_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocess_obj.transform(input_feature_test_data)

             # Apply preprocessor object on our train data and test data
            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

            save_object(file_path=self.transformation_config.preprocess_obj_file_path,
                        obj=preprocess_obj)

            return (train_array,
                    test_array,
                    self.transformation_config.preprocess_obj_file_path)

        except Exception as e:
            raise CustomException(e)



if __name__ =="__main__":
    data_transformation= DataTransformation()
    data_transformation.initiateDataTransformation(
        train_path= os.path.join("artifacts/data_ingestion", 'train.csv'),
        test_path= os.path.join("artifacts/data_ingestion", 'test.csv')
    )