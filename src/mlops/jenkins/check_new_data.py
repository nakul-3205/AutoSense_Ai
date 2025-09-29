import pymongo
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION")

TRACK_FILE = "last_count.txt"

def get_current_count():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection.count_documents({})

def get_previous_count():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def update_count(count):
    with open(TRACK_FILE, "w") as f:
        f.write(str(count))

if __name__ == "__main__":
    current_count = get_current_count()
    previous_count = get_previous_count()
    temp_count=current_count-previous_count
    print(f"Previous count: {previous_count}, Current count: {current_count}")

    if temp_count>10000:
        print(" New data detected! Updating record...")
        update_count(current_count)
        exit(0)  # Jenkins proceeds
    else:
        print(" No new data found. Skipping pipeline.")
        exit(1)  # Jenkins stops here
