import os

from pymongo import MongoClient
from pymongo.errors import PyMongoError

MONGO_URI = os.environ.get("STATSTOCK_MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("STATSTOCK_DB", "StockDatabase")
OUTPUT_DIR = os.environ.get("STATSTOCK_OUTPUT", "output")


def get_database():
    """Return the project database, failing early if MongoDB is unreachable."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        # MongoClient connects lazily, so ping to find out whether it really works.
        client.admin.command("ping")
    except PyMongoError as e:
        raise RuntimeError(f"Could not reach MongoDB at {MONGO_URI}: {e}") from e
    return client[DB_NAME]
