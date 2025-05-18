import pandas as pd
import html
import re
from tqdm.auto import tqdm
from collections import defaultdict
from config import Config

class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self):
        self.df = None
        self.review_texts = []
        self.review_meta = []
        self.drug_condition_reviews = defaultdict(list)
        self.drug_mapping = self._create_drug_mapping()
    
    def _create_drug_mapping(self):
        """Create brand ↔ generic drug mapping"""
        mapping = {
            "Montelukast": "Singulair",
            "Fluticasone / salmeterol": "Advair",
            "Isotretinoin": "Accutane",
            "Claravis": "Isotretinoin",
            "Valacyclovir": "Valtrex",
            "Sertraline": "Zoloft",
            "Fluoxetine": "Prozac",
            "Escitalopram": "Lexapro",
            "Ibuprofen": "Advil",
            "Acetaminophen": "Tylenol",
            "Cetirizine": "Zyrtec",
        }
        # Add reverse mappings
        mapping.update({v: k for k, v in mapping.items()})
        return mapping
    
    def load_drug_data(self, file_path: str = None):
        """
        Load and sample drug review data
        
        Args:
            file_path: Path to CSV file (uses config default if None)
        """
        file_path = file_path or Config.DRUG_DATA_PATH
        
        print("Loading drug review data...")
        
        try:
            self.df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Please update the file path in config.py")
            return False
        
        # Sample and clean data
        self.df = (
            self.df[["drugName", "condition", "review", "rating", "usefulCount", "date"]]
            .sample(n=min(Config.SAMPLE_SIZE, len(self.df)), random_state=Config.RANDOM_STATE)
            .reset_index(drop=True)
        )
        
        # Clean drug and condition names
        self.df["drugName"] = self.df["drugName"].str.strip()
        self.df["condition"] = self.df["condition"].str.strip()
        
        print(f"Loaded {len(self.df)} drug reviews")
        return True
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not text:
            return ""
        
        # Unescape HTML entities
        txt = html.unescape(text)
        
        # Normalize whitespace
        txt = re.sub(r"\s+", " ", txt).strip()
        
        # Add space after punctuation if missing
        txt = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", txt)
        
        # Remove quotes
        txt = re.sub(r"[\"']", "", txt)
        
        return txt
    
    @staticmethod
    def chunk_text(text: str, max_chars: int = None) -> list:
        """
        Split text into chunks of specified size
        
        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        max_chars = max_chars or Config.MAX_CHUNK_CHARS
        txt = DataProcessor.clean_text(text)
        
        if not txt:
            return []
        
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        
        chunks = []
        current = ""
        
        for sentence in sentences:
            separator = ". " if current and not current.endswith((".", "!", "?", " ")) else ""
            
            if len(current) + len(separator) + len(sentence) <= max_chars:
                current += separator + sentence
            else:
                # Finish current chunk
                if not current.endswith((".", "!", "?")):
                    current += "."
                chunks.append(current)
                current = sentence
        
        # Add final chunk
        if current:
            if not current.endswith((".", "!", "?")):
                current += "."
            chunks.append(current)
        
        return chunks
    
    def process_reviews(self):
        """Process reviews into chunks and build metadata"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_drug_data first.")
        
        print("Processing reviews into chunks...")
        
        self.review_texts = []
        self.review_meta = []
        self.drug_condition_reviews = defaultdict(list)
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Chunking"):
            # Skip rows with missing data
            if pd.isna(row.review) or pd.isna(row.drugName) or pd.isna(row.condition):
                continue
            
            # Process each chunk
            for chunk in self.chunk_text(row.review):
                # Prepend condition to chunk for better embeddings
                self.review_texts.append(f"{row.condition} | {chunk}")
                
                # Create metadata
                meta = {
                    "drug": row.drugName,
                    "condition": row.condition,
                    "text": chunk,
                    "rating": row.rating,
                    "useful": row.usefulCount
                }
                self.review_meta.append(meta)
                self.drug_condition_reviews[(row.drugName, row.condition)].append(meta)
        
        print(f"Created {len(self.review_texts)} review chunks")
        return self.review_texts, self.review_meta, self.drug_condition_reviews
    
    def get_conditions(self) -> list:
        """Get list of unique conditions"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_drug_data first.")
        
        return self.df["condition"].dropna().unique().tolist()
    
    def get_top_reviews(self, drug: str, condition: str, n: int = None) -> list:
        """
        Get top-rated reviews for a drug-condition pair
        
        Args:
            drug: Drug name
            condition: Medical condition
            n: Number of reviews to return
            
        Returns:
            List of top review metadata
        """
        n = n or Config.TOP_REVIEWS_COUNT
        
        # Try exact match first
        reviews = self.drug_condition_reviews.get((drug, condition), [])
        
        # Try with drug mapping if no exact match
        if not reviews and drug in self.drug_mapping:
            reviews = self.drug_condition_reviews.get(
                (self.drug_mapping[drug], condition), []
            )
        
        # Sort by weighted score (rating + usefulness)
        return sorted(
            reviews,
            key=lambda r: (r["rating"] * 0.6 + r["useful"] * 0.4),
            reverse=True
        )[:n]