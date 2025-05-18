import torch
from unsloth import FastLanguageModel
from transformers import pipeline
from config import Config

class UnslothLLMManager:
    """Manages the Unsloth LLM model for text generation"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self._load_model()
    
    def _load_model(self):
        """Load the Mistral model with Unsloth optimizations"""
        print(f"Loading model: {Config.MODEL_ID}")
        
        # Load model with 4-bit quantization
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=Config.MODEL_ID,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            dtype=Config.DTYPE,
            load_in_4bit=Config.LOAD_IN_4BIT,
        )
        
        # Set up text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE,
            do_sample=Config.DO_SAMPLE,
        )
        
        print("Model loaded successfully!")
    
    def generate_text(self, prompt: str, max_tokens: int = None) -> str:
        """
        Generate text using the loaded model
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (overrides config if provided)
            
        Returns:
            Generated text string
        """
        # Format prompt for Mistral
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Set max tokens
        max_new_tokens = max_tokens or Config.MAX_NEW_TOKENS
        
        try:
            # Generate response
            response = self.pipe(
                formatted_prompt,
                return_full_text=False,
                max_new_tokens=max_new_tokens,
                temperature=Config.TEMPERATURE,
                do_sample=Config.DO_SAMPLE,
            )[0]["generated_text"]
            
            return response.strip()
        
        except Exception as e:
            print(f"Error generating text: {e}")
            return "Error: Could not generate response"
    
    def generate_summary(self, prompt: str) -> str:
        """Generate a medical summary using default settings"""
        return self.generate_text(prompt)
    
    def is_loaded(self) -> bool:
        """Check if model is properly loaded"""
        return self.model is not None and self.tokenizer is not None and self.pipe is not None