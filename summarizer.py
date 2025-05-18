"""
Drug Summary Generation Module for DrugBot

This module handles the generation of drug summaries using LLM and review data.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import pandas as pd
from .fda_manager import FDAManager


class DrugSummarizer:
    """Handles drug summary generation using LLM and review data."""
    
    def __init__(self, llm_pipeline, drug_condition_reviews: Dict, drug_mapping: Dict):
        """
        Initialize the drug summarizer.
        
        Args:
            llm_pipeline: The language model pipeline for generation
            drug_condition_reviews: Dictionary of drug-condition review data
            drug_mapping: Dictionary for drug name mapping (brand <-> generic)
        """
        self.pipe = llm_pipeline
        self.drug_condition_reviews = drug_condition_reviews
        self.drug_mapping = drug_mapping
    
    def get_top_reviews(self, drug: str, condition: str, n: int = 10) -> List[Dict]:
        """
        Get top reviews for a drug-condition pair.
        
        Args:
            drug: Drug name
            condition: Medical condition
            n: Number of top reviews to return
            
        Returns:
            List of top review dictionaries
        """
        revs = self.drug_condition_reviews.get((drug, condition), [])
        if not revs and drug in self.drug_mapping:
            revs = self.drug_condition_reviews.get(
                (self.drug_mapping[drug], condition), []
            )
        return sorted(
            revs,
            key=lambda r: (r["rating"]*0.6 + r["useful"]*0.4),
            reverse=True
        )[:n]
    
    @staticmethod
    def clean_drug_name(drug_name: str) -> str:
        """
        Clean a drug name by removing common suffixes and extracting
        the first component if it contains a slash (/).
        
        Args:
            drug_name: Raw drug name
            
        Returns:
            Cleaned drug name
        """
        if not drug_name:
            return ""

        # If there's a slash, extract the first component
        if "/" in drug_name:
            primary_component = drug_name.split("/")[0].strip()
            return primary_component

        return drug_name
    
    def summarize_drug(self, drug: str, condition: str, fda_manager: FDAManager) -> Tuple[str, List[Dict], float, bool]:
        """
        Generate a comprehensive drug summary using LLM.
        
        Args:
            drug: Drug name
            condition: Medical condition
            fda_manager: FDA data manager instance
            
        Returns:
            Tuple of (summary, examples, avg_rating, fda_found)
        """
        # Get reviews
        reviews = self.get_top_reviews(drug, condition, 15)
        if not reviews:
            return f"No reviews found for {drug} treating {condition}.", [], 0, False

        # Calculate rating
        avg_rating = sum(r["rating"] for r in reviews) / len(reviews)

        # Get review snippets
        snippets = [r["text"] for r in reviews][:8]
        reviews_text = "".join([f"Review {i+1}: {s}\n" for i, s in enumerate(snippets)])

        # Get FDA info
        fda_text, fda_found = fda_manager.get_drug_fda_info(drug)

        # Create effectiveness instruction based on FDA data availability
        if fda_found:
            effectiveness_instruction = f"[Compare patient experiences with FDA data: {fda_text}. Analyze effectiveness for {condition}, highlighting matches and differences between official uses and patient reports.]"
        else:
            effectiveness_instruction = f"[Analyze reported effectiveness for {condition} based solely on patient reviews. Focus on specific symptoms relieved or not relieved.]"

        # Create prompt for Mistral format
        prompt = f"""Analyze these patient reviews and FDA data for {drug} used in treating {condition}:

DRUG INFORMATION:
  Name: {drug}
  Medical Condition: {condition}
  Patient Rating: {avg_rating:.1f}/10 (based on patient reviews)

FDA OFFICIAL DATA:
{fda_text}

PATIENT EXPERIENCES:
{reviews_text}

Based on this information, provide a comprehensive analysis with:

EFFECTIVENESS:{effectiveness_instruction}

SIDE EFFECTS:
[List the common side effects mentioned by patients]

PROS AND CONS:
[List key benefits and drawbacks of this medication according to patients]

SUMMARY:
[Write 2-3 sentences summarizing the overall patient experience]
"""

        # Format for Mistral
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        # Generate summary
        try:
            response = self.pipe(
                formatted_prompt,
                return_full_text=False,
                max_new_tokens=500,
                temperature=0.1,
                do_sample=True,
            )[0]["generated_text"]

            summary = response.strip()

            # Fallback if generation fails
            if not summary or len(summary) < 50:
                summary = f"Based on patient reviews, {drug} appears to have mixed effectiveness for treating {condition}, with an average rating of {avg_rating:.1f}/10."

        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = f"Analysis of {drug} for {condition}: Average rating {avg_rating:.1f}/10."

        # Get example reviews
        examples = [
            {"text": r["text"], "rating": r["rating"], "useful": r["useful"]}
            for r in reviews[:3]
        ]

        return summary, examples, avg_rating, fda_found