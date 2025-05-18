"""
FDA Data Management Module for DrugBot

This module handles FDA data loading, normalization, and information retrieval.
"""

import pandas as pd
import re
from typing import Tuple


class FDAManager:
    """Manages FDA drug information and provides search functionality."""
    
    def __init__(self, fda_data_path: str = None):
        """
        Initialize the FDA manager.
        
        Args:
            fda_data_path: Path to the FDA CSV data file
        """
        self.fda_df = None
        if fda_data_path:
            self.load_fda_data(fda_data_path)
    
    def load_fda_data(self, fda_data_path: str) -> None:
        """
        Load FDA data from CSV file.
        
        Args:
            fda_data_path: Path to the FDA CSV data file
        """
        try:
            self.fda_df = pd.read_csv(fda_data_path)
            print(f"Loaded FDA database with {len(self.fda_df)} records")
        except Exception as e:
            print(f"Error loading FDA data: {e}")
            self.fda_df = None
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for better matching.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text string
        """
        text = str(text).lower()
        text = re.sub(r'\([^)]*\)|\[.*?\]|[^a-z0-9\s]', '', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'\b(mg|mcg|g|ml|l|iu|%)\b', '', text)
        return ' '.join(text.split()).strip()
    
    def get_drug_fda_info(self, drug: str) -> Tuple[str, bool]:
        """
        Find FDA information for a drug using regex-based matching.
        
        Args:
            drug: Drug name to search for
            
        Returns:
            Tuple of (fda_text, fda_found)
        """
        # Handle empty inputs
        if self.fda_df is None or self.fda_df.empty:
            return "No FDA data available for this medication.", False

        # Columns to search
        fda_columns = [
            'openfda.brand_name',
            'openfda.generic_name',
            'openfda.active_ingredient',
            'use_in_specific_populations',
            'warnings_and_cautions',
            'contraindications',
            'adverse_reactions',
            'do_not_use',
            'when_using',
            'ask_doctor'
        ]
        fda_columns = [c for c in fda_columns if c in self.fda_df.columns]
        if not fda_columns:
            return "No FDA data available for this medication.", False

        norm_drug = self.normalize_text(drug)
        if not norm_drug:
            return "No FDA data available for this medication.", False

        # Prepare regex patterns
        p = re.escape(norm_drug)
        raw_patterns = [
            rf'^{p}\b',     # starts with
            rf'\b{p}$',     # ends   with
            rf'\b{p}\b',    # exact word
            rf'{p}',        # substring
            rf'\b{p}\w*\b', # prefix
            rf'\b\w*{p}\b', # suffix
        ]
        compiled = [re.compile(pat, re.IGNORECASE) for pat in raw_patterns]

        # Search through FDA data
        matches = []
        for idx, row in self.fda_df.iterrows():
            for col in fda_columns:
                val = row.get(col)
                if pd.isna(val):
                    continue

                # 1) Collapse list→string
                if isinstance(val, list):
                    col_text = " ".join(str(x) for x in val if x)
                else:
                    col_text = str(val)
                col_norm = self.normalize_text(col_text)

                # 2) Test any regex
                if any(cp.search(col_norm) for cp in compiled):
                    matches.append((idx, row, col))
                    break  # move to next row as soon as we find one match

            if matches:
                break  # stop after first matching row

        # If no matches, try a fallback substring-only scan
        if not matches:
            flex_pat = re.compile(rf'\w*{p}\w*', re.IGNORECASE)
            for idx, row in self.fda_df.iterrows():
                for col in fda_columns:
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    if isinstance(val, list):
                        col_text = " ".join(str(x) for x in val if x)
                    else:
                        col_text = str(val)
                    if flex_pat.search(self.normalize_text(col_text)):
                        matches.append((idx, row, col))
                        break
                if matches:
                    break

        if not matches:
            return "No FDA data available for this medication.", False

        # We have at least one match in matches[0]
        _, best_row, best_col = matches[0]

        # Build output sections
        important_fields = [
            'openfda.brand_name',
            'openfda.generic_name',
            'openfda.active_ingredient',
            'use_in_specific_populations',
            'warnings_and_cautions',
            'contraindications',
            'adverse_reactions',
            'do_not_use',
            'when_using',
            'ask_doctor'
        ]
        sections = [f"Matched Field: {best_col}"]
        for field in important_fields:
            if field in best_row and not pd.isna(best_row[field]):
                val = best_row[field]
                if isinstance(val, list):
                    val = " ".join(val)
                val = str(val)
                # Truncate very long text
                if len(val) > 300:
                    val = val[:300] + "..."
                sections.append(f"{field.replace('_',' ').title()}: {val}")

        return "\n".join(sections), True