from pymongo import MongoClient

def get_database():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["StockDatabase"]
        print("✅ MongoDB connection successful!")
        return db
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        raise