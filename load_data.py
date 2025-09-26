import os
import json
import sys
from dotenv import load_dotenv
import pandas as pd 
import pymongo
import numpy as np
from src.utils.logging import logging
from src.utils.exception import CustomException


load_dotenv()

MONGO_URL=os.getenv('MONGODB_URI')
import certifi
ca=certifi.where()


class DataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e :
            raise CustomException(e,sys)
    
    def csv_to_json(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            # records=json.loads((data.T.to_json()).values())
            records = json.loads(data.T.to_json())
            records = list(records.values())  

            return records
        except Exception as e :
            raise CustomException(e,sys)
    
    def load_in_mongo(self,records,collection,database):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_client=pymongo.MongoClient(MONGO_URL)
            self.database=self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            logging.info('sucessfull connect to database')

            return( len(self.records))

        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    FILE_PATH="Network_Data/phisingData.csv"
    DATABASE="SCANIFYAI_DATABASE"
    COLLECTION="SCANIFY_AI"
    networkobj=DataExtract()
    record=networkobj.csv_to_json(file_path=FILE_PATH)
    no_of_records=networkobj.load_in_mongo(record,DATABASE,COLLECTION)
    print(no_of_records)
    
logging.info('sucessfully loaded the data in mongo')



