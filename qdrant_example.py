import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load variables from .env into the system environment
load_dotenv()

# Access a variable
qdrant_url = os.getenv("qdrant_url")
qdrant_api_key = os.getenv("qdrant_api_key")


qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_api_key,
)

print(qdrant_client.get_collections())