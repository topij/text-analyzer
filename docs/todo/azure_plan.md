# Azure ML Migration Plan

## 1. Environment Setup
1. Create Azure ML environment with required dependencies
   - Create environment.yml for Azure ML
   ```yaml
   name: semantic-analyzer
   channels:
     - conda-forge
   dependencies:
     - python=3.9
     - pip
     - pip:
       - azure-ml
       - azure-storage-blob
       - voikko-fi  # Finnish language support
       - nltk
       - pandas
       - pydantic
       - langchain
       - langchain-openai
       - langchain-anthropic
   ```
   - Register environment in Azure ML workspace

## 2. Data Storage Migration
1. Set up Azure Blob Storage container for:
   - Parameter files (parameters_en.xlsx, parameters_fi.xlsx)
   - Test data files
   - Configuration files (config.yaml, language_processing.yaml)

2. Modify FileUtils to support Azure Storage
   - Add Azure Blob Storage support to save_data_to_disk
   - Add Azure path handling
   - Add connection string/credentials handling

## 3. Notebook Migration
1. Add Azure ML setup code at start of notebook:
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<sub_id>",
    resource_group_name="<rg>",
    workspace_name="<workspace>"
)

# Initialize storage client
blob_service_client = BlobServiceClient.from_connection_string("<conn_string>")
```

2. Modify data loading code to use Azure Storage
   - Update parameter file paths
   - Update test data paths
   - Use blob storage client for file operations

3. Configure compute target
   - Create compute configuration
   - Select appropriate VM size
   - Set up autoscaling if needed

## 4. Testing Steps
1. Test basic functionality
   - Test parameter loading from blob storage
   - Test file saving to blob storage
   - Verify compute resources

2. Test full analysis pipeline
   - Run complete analysis on test data
   - Verify results match local execution
   - Check performance metrics

## 5. Optimization
1. Cache frequently accessed files
2. Use mounted storage when possible
3. Configure optimal compute resources
4. Add logging to Azure Monitor

## 6. Documentation
1. Update README with Azure ML setup instructions
2. Document new storage paths and configurations
3. Add Azure-specific troubleshooting guide

## Next Steps After Notebook Migration
1. Create reusable Azure ML pipeline components
2. Set up CI/CD for Azure deployment
3. Add monitoring and alerting
4. Create Azure ML endpoints for production use