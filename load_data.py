import pandas as pd
from mongo import connect_to_dbs

db_client = connect_to_dbs()
db = db_client['AutoSense']
collection = db['data']

df = pd.read_csv("data/raw/data.csv")

collection.insert_many(df.to_dict('records'))

print("File uploaded to MongoDB successfully!")
