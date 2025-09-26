import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL")

if not mongo_db_url:
    raise ValueError("MongoDB URL not found in environment variables!")

_cached = {"conn": None}

def connect_to_dbs():
    global _cached
    if _cached["conn"]:
        return _cached["conn"]

    try:
        client = MongoClient(mongo_db_url, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("Database connected")
        _cached["conn"] = client
        return _cached["conn"]
    except Exception as e:
        _cached["conn"] = None
        raise e
