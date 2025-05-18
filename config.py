import os
import torch

# Configuration constants
class Config:
    # Model settings
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    MAX_SEQ_LENGTH = 2048
    DTYPE = torch.float16
    LOAD_IN_4BIT = True
    
    # Text generation settings
    MAX_NEW_TOKENS = 300
    TEMPERATURE = 0.1
    DO_SAMPLE = True
    
    # Embedding settings
    EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
    BATCH_SIZE_EMBED = 256
    BATCH_SIZE_COND = 128
    
    # Text processing settings
    MAX_CHUNK_CHARS = 500
    TOP_K_ALTERNATIVES = 5
    TOP_REVIEWS_COUNT = 10
    
    # Data paths (modify these for your setup)
    DRUG_DATA_PATH = "/content/drive/MyDrive/Drug_data/drugsComTrain_raw.csv"
    FDA_DATA_PATH = "/content/drive/MyDrive/Drug_data/combined_all.csv"
    
    # Sampling settings
    SAMPLE_SIZE = 60000
    RANDOM_STATE = 42
    
    # Visualization settings
    FIGURE_SIZE = (12, 17)
    DPI = 300
    
    @classmethod
    def check_cuda(cls):
        """Check CUDA availability and print GPU info"""
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    @classmethod
    def setup_cuda(cls):
        """Setup CUDA environment"""
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return "cuda" if torch.cuda.is_available() else "cpu"