import os

from ai_code_detector.config import PROJECT_ROOT
from ai_code_detector.models.code_embedder import CodeEmbeddingEncoder

if __name__ == "__main__":

    save_path = os.path.join(PROJECT_ROOT, "embeddings", "sf_coder_400membeddings.pkl")
    encoder = CodeEmbeddingEncoder(
        model_name="Salesforce/SFR-Embedding-Code-400M_R",
        max_length=8192,
        cache_dir="cache",
        device="cuda",
        pooling_strategy="cls"
    )

    encoder.encode_and_save(texts=["print('Hello, world!')"], save_path=save_path)