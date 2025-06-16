import os
import sys
from azure.storage.blob import BlobServiceClient

def test_connection(connection_string):
    try:
        print("Connecting to Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        print("Connection successful!\nListing containers:")
        containers = blob_service_client.list_containers()

        for container in containers:
            print(f" - {container['name']}")

        print("Test completed successfully.")
    except Exception as e:
        print(f"Failed to connect or list containers: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python python.py <AZURE_STORAGE_CONNECTION_STRING>")
    else:
        connection_string = sys.argv[1]
        test_connection(connection_string)
