import os
from dotenv import load_dotenv
from FileUtils import FileUtils
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def setup_environment():
    """Setup environment variables and connections."""
    # Load environment variables
    load_dotenv('.env')
    
    # Initialize FileUtils with Azure support
    file_utils = FileUtils.create_azure_utils(
        connection_string=os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    )
    
    return file_utils

def init_azure_ml():
    """Initialize Azure ML workspace."""
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZURE_ML_WORKSPACE")
    )
    return ml_client