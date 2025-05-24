import os

import pandas as pd

from ai_code_detector.config import PROJECT_ROOT
from ai_code_detector.models.code_embedder import CodeEmbeddingEncoder

if __name__ == "__main__":
    
    # Load the dataset from the data directory
    data_path = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
    print(f"Loading dataset from: {data_path}")
    
    # Read the CSV file
    df = pd.read_csv(data_path)
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Extract the code column
    code_texts = df['code'].tolist()
    print(f"Extracted {len(code_texts)} code samples")
    
    # Initialize the encoder
    save_path = os.path.join(PROJECT_ROOT, "embeddings", "dataset_embeddings.pkl")
    encoder = CodeEmbeddingEncoder(
        model_name="Salesforce/SFR-Embedding-Code-400M_R",
        max_length=8192,
        cache_dir="cache",
        device="cuda",
        pooling_strategy="cls"
    )
    
    print("Generating embeddings for all code samples...")
    encoder.encode_and_save(texts=code_texts, save_path=save_path, batch_size=64, show_progress=True)
    print(f"Embeddings saved to: {save_path}")