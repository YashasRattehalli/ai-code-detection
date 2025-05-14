import os

import pandas as pd
from huggingface_hub import login
from huggingface_hub.hf_api import HfApi


def download_dataset(dataset_path, 
                     token=None, output_dir="./data"):
    """
    Download a dataset from HuggingFace.
    
    Args:
        dataset_path (str): Path to the dataset in Huggingface Hub
        token (str): HuggingFace API token. If None, will use the token from the environment.
        output_dir (str): Directory to save the downloaded dataset
        
    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Login to HuggingFace if token is provided
    if token:
        login(token=token)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the dataset path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[5:]
    
    # Construct the full path using HF format
    if not dataset_path.startswith("datasets/"):
        dataset_path = f"datasets/{dataset_path}"
    
    # Download and read the dataset
    try:
        # Direct read using pandas
        df = pd.read_parquet(f"hf://{dataset_path}")
        
        # Save locally if output_dir is specified
        local_path = os.path.join(output_dir, os.path.basename(dataset_path))
        df.to_parquet(local_path)
        
        print(f"Dataset downloaded and saved to {local_path}")
        return df
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Alternative manual download using HfApi if direct read fails
        try:
            api = HfApi()
            file_info = api.hf_hub_download(
                repo_id="/".join(dataset_path.split("/")[:2]),
                filename="/".join(dataset_path.split("/")[2:]),
                local_dir=output_dir
            )
            df = pd.read_parquet(file_info)
            print(f"Dataset downloaded using alternative method and saved to {file_info}")
            return df
        except Exception as e2:
            print(f"Alternative download method also failed: {e2}")
            raise

if __name__ == "__main__":
    # Example usage
    # Note: Set your token or login using huggingface-cli login first
    #df = download_dataset("datasets/DaniilOr/CoDET-M4/dataset_without_comments.parquet",
    #                     token="dummy")
    #print(f"Dataset shape: {df.shape}")
    #print(df.head())
    #df.to_csv("dataset_without_comments.csv", index=False)  
    # Load dataset
    df = pd.read_csv("dataset.csv")
    df.head(100).to_csv("dataset_head.csv", index=False)
