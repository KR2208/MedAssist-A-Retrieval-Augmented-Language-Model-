"""
Evaluation Module for DrugBot

This module handles the evaluation of drug summaries using LLM.
"""

import re
from typing import List, Tuple, Dict, Any


class SummaryEvaluator:
    """Evaluates the quality of drug summaries using LLM."""
    
    def __init__(self, llm_pipeline):
        """
        Initialize the summary evaluator.
        
        Args:
            llm_pipeline: The language model pipeline for evaluation
        """
        self.pipe = llm_pipeline
    
    def evaluate_drug_summaries(self, summaries_list: List[Tuple]) -> List[Dict]:
        """
        Evaluate a list of drug summaries after they've all been generated.

        Args:
            summaries_list: A list of tuples containing (drug, condition, summary, avg_rating, fda_found)

        Returns:
            A list of evaluation results
        """
        evaluations = []

        for drug, condition, summary, avg_rating, fda_found in summaries_list:
            # Create evaluation prompt for the LLM
            evaluation_prompt = f"""<s>[INST]
I need you to evaluate the quality of a drug summary. Here's the original data:

DRUG: {drug}
CONDITION: {condition}
AVERAGE RATING: {avg_rating:.1f}/10
FDA DATA AVAILABLE: {"Yes" if fda_found else "No"}

And here's the generated summary:
---
{summary}
---

Please evaluate this summary on the following criteria:
1. Completeness (1-10): Does it cover all required sections (Effectiveness, Side Effects, Pros/Cons, Summary)?
2. Accuracy (1-10): Does the summary accurately reflect the patient rating and experiences?
3. Usefulness (1-10): How helpful would this be to someone considering this medication?
4. Information Use (1-10): How effectively does the summary use all available information (including FDA data if available)?
5. Readability (1-10): Is it well-organized and easy to understand?

For each criterion, give a score from 1-10 and a brief explanation.

Format your response as follows:
COMPLETENESS: [score]
ACCURACY: [score]
USEFULNESS: [score]
INFORMATION_USE: [score]
READABILITY: [score]

SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]
[/INST]"""

            try:
                eval_response = self.pipe(
                    evaluation_prompt,
                    return_full_text=False,
                    max_new_tokens=1000,
                    temperature=0.1,
                    do_sample=True,
                )[0]["generated_text"]

                # Create evaluation result
                evaluation = {
                    "drug": drug,
                    "condition": condition,
                    "suggestions": []
                }

                # Extract criterion scores
                criteria = {
                    "COMPLETENESS": "completeness",
                    "ACCURACY": "accuracy",
                    "USEFULNESS": "usefulness",
                    "INFORMATION_USE": "information_use",
                    "READABILITY": "readability"
                }

                for criterion, key in criteria.items():
                    match = re.search(rf'{criterion}: (\d+)', eval_response)
                    if match:
                        evaluation[key] = int(match.group(1))

                # Extract suggestions
                suggestions_section = re.search(r'SUGGESTIONS:(.*?)$', eval_response, re.DOTALL)
                if suggestions_section:
                    suggestion_text = suggestions_section.group(1).strip()
                    suggestions = re.findall(r'-\s+(.*?)(?=$|[\n\r]-\s+)', suggestion_text, re.DOTALL)
                    evaluation["suggestions"] = [s.strip() for s in suggestions]

                evaluations.append(evaluation)

            except Exception as e:
                evaluations.append({
                    "drug": drug,
                    "condition": condition,
                    "error": str(e)
                })

        return evaluations