"""MongoDB connector for data ingestion."""

import os
from typing import Any

import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.errors import ServerSelectionTimeoutError

from dotenv import load_dotenv

from config.logging_config import LoggingConfig


logger = LoggingConfig().get_logger(__name__)

if not os.getenv("MONGODB_URL"):
    load_dotenv()


class MongoDBConnector:
    """MongoDB connector for data ingestion."""

    def __init__(
            self,
            connection_string: str,
            database_name: str,
            collection_name: str
    ):
        """Initialize MongoDB connector.

        Args:
            connection_string: MongoDB connection URL
            database_name: Name of the database
            collection_name: Name of the collection
        """
        if not connection_string:
            raise ValueError(
                "MongoDB connection string is empty."
            )

        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.database = None
        self.collection = None

        logger.info(
            f"Initializing MongoDB connector for {
                database_name
            }.{collection_name}"
        )

    def connect(self) -> None:
        """Establish MongoDB connection."""
        try:
            self.client = MongoClient(
                self.connection_string,
                server_api=ServerApi('1'),
                serverSelectionTimeoutMS=10000,  # 10 seconds
                connectTimeoutMS=10000,
                socketTimeoutMS=30000,
                retryWrites=True,
                w='majority'
            )

            # Test connection
            self.client.admin.command("ping")

            # Get database and collection
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]

            logger.info(
                f"Successfully connected to MongoDB: {
                    self.database_name
                }.{
                    self.collection_name
                }"
            )

        except ServerSelectionTimeoutError as e:
            logger.error(
                f"MongoDB connection timeout. Check your connection: {e}"
            )
            raise ConnectionError(f"Cannot connect to MongoDB: {e}")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise ConnectionError(f"MongoDB connection failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise

    def test_connection(self) -> bool:
        """Test if MongoDB connection is active.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            if self.client is None:
                logger.error("MongoDB client is not initialized")
                return False

            self.client.admin.command('ping')
            logger.info("MongoDB connection test successful")
            return True

        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False

    def fetch_data(
        self,
        query: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch data from MongoDB collection.

        Args:
            query: MongoDB query filter
            projection: Fields to include/exclude
            limit: Maximum number of documents to fetch

        Returns:
            pandas.DataFrame: Fetched data
        """
        try:
            if self.collection is None:
                raise ConnectionError("Connection may have failed.")

            # Build query
            query = query or {}
            logger.info(f"Fetching data with query: {query}, limit: {limit}")

            # Build cursor
            cursor = self.collection.find(query, projection)

            if limit:
                cursor = cursor.limit(limit)

            # Convert to dataframe
            data = list(cursor)

            if not data:
                logger.warning("No data found in MongoDB collection")
                return pd.DataFrame(data)

            df = pd.DataFrame(data)

            # Remove _id field
            if "_id" in df.columns:
                df = df.drop("_id", axis=1)

            logger.info(f"Successfully fetched {len(df)} rows from MongoDB")
            return df

        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data fetch: {e}")
            raise

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dict containing collection stats
        """
        try:
            if self.collection is None:
                raise ConnectionError("Collection not initialized")

            # Get document count
            doc_count = self.collection.count_documents({})

            info = {
                "document_count": doc_count,
                "collection": self.collection.name,
                "database_name": self.database_name
            }

            # Get sample document structure
            if doc_count > 0:
                sample_doc = self.collection.find_one()
                if sample_doc:
                    fields = [k for k in sample_doc.keys() if k != "_id"]
                    info["fields"] = fields
                    info["field_count"] = len(fields)

            logger.info(f"Collection info: {info}")
            return info

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            self.collection = None
            logger.info("MongoDB connection closed")

    def __enter__(self):
        """Context manager entry.

        Returns:
            self: The connector instance
        """
        if self.client is None:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed.

        Args:
            exc_type: Exception type (if an exception occurred)
            exc_val: Exception value (the exception instance)
            exc_tb: Exception traceback

        Returns:
            bool: False to propagate exceptions (default behavior)
        """
        # Log if there was an exception
        if exc_type is not None:
            logger.warning(
                f"Exception occurred during MongoDB operations: {
                    exc_type.__name__}: {exc_val}"
            )

        # Always close the connection
        self.close()

        # Return False to propagate the exception (if any)
        # Return True would suppress the exception
        return False
