"""
main.py - Main application entry point for DrugBot
"""

import os
import pandas as pd
from data_processor import DataProcessor
from model_manager import ModelManager
from drug_search import DrugSearch
from summarizer import DrugSummarizer
from evaluation import DrugEvaluator
from visualizer import visualize_drug_evaluations, display_header


def main():
    """
    Interactive main function for DrugBot with automatic evaluation
    """
    # Initialize components
    print("Initializing DrugBot...")
    
    # Load data
    data_processor = DataProcessor()
    df = data_processor.load_and_prepare_data('/path/to/drugsComTrain_raw.csv')
    
    # Initialize model
    model_manager = ModelManager()
    model, tokenizer, pipe = model_manager.load_model()
    
    # Initialize search and summarization
    drug_search = DrugSearch(df, model_manager.embeddings)
    drug_search.build_indexes()
    
    # Initialize summarizer and evaluator
    summarizer = DrugSummarizer(drug_search, pipe)
    evaluator = DrugEvaluator(pipe)
    
    # Load FDA data
    fda_df = pd.read_csv("/path/to/combined_all.csv")
    print(f"Loaded FDA database with {len(fda_df)} records")

    display_header("👋 DrugBot: AI-Powered Medication Assistant")
    print("Tell me about your health condition, and I'll suggest medications.\n")

    # Keep track of all drugs and summaries for later evaluation
    all_summaries = []

    while True:
        query = input("You (type 'exit' to quit, 'evaluate' to evaluate all results): ").strip()

        if query.lower() == 'exit':
            print("Goodbye! Stay healthy!")
            break

        if query.lower() == 'evaluate':
            if not all_summaries:
                print("No summaries to evaluate yet. Try asking about a condition first.")
                continue

            # Group summaries by condition
            conditions = {}
            for drug, condition, summary, avg_rating, fda_found in all_summaries:
                if condition not in conditions:
                    conditions[condition] = []
                conditions[condition].append((drug, condition, summary, avg_rating, fda_found))

            # Process each condition separately
            for condition, summaries in conditions.items():
                # Evaluate summaries
                print(f"\nEvaluating medications for {condition}...")
                evaluations = evaluator.evaluate_drug_summaries(summaries)

                # Create visualization
                print(f"Generating visualization for {condition} medications...")
                viz_file = visualize_drug_evaluations(evaluations, condition)
                if viz_file:
                    print(f"Visualization saved to {viz_file}")
                else:
                    print(f"No valid evaluation data for {condition}")

            continue

        try:
            # Get condition and suggested drugs
            cond, drugs = drug_search.get_alternatives(query)
            print(f"\n🩺 Detected condition: {cond}\n")

            if not drugs:
                print("No medications found for this condition. Please try describing your symptoms differently.")
                continue

            print(f"Found {len(drugs)} potential medications for {cond}")

            # Current batch summaries
            batch_summaries = []

            # Process each drug
            for i, drug in enumerate(drugs, 1):
                print(f"Processing {i}/{len(drugs)}: {drug}")
                primary_drug = data_processor.clean_drug_name(drug)

                # Generate summary
                summary, examples, avg_rating, fda_found = summarizer.summarize_drug(
                    primary_drug, cond, fda_df
                )

                # Store summary for later evaluation
                batch_summaries.append((primary_drug, cond, summary, avg_rating, fda_found))
                all_summaries.append((primary_drug, cond, summary, avg_rating, fda_found))

                # Display header
                display_header(f"{drug} for {cond}")

                # Display medication overview
                print(f"\n💊 MEDICATION OVERVIEW:")
                print(f"  Drug: {drug}")
                print(f"  Condition: {cond}")
                print(f"  Average Rating: {avg_rating:.1f}/10")
                print(f"  FDA Data: {'Available' if fda_found else 'Not Available'}")

                # Display summary
                print("\n📊 PATIENT REVIEW SUMMARY:\n")
                print(summary)

                # Display sample reviews
                print("\n📝 SAMPLE PATIENT REVIEWS:")
                for ex in examples:
                    print(f'• "{ex["text"]}" (Rating: {ex["rating"]}/10, Votes: {ex["useful"]})')

                # Add disclaimer
                print("\n⚠️ DISCLAIMER: This information is for educational purposes only and not intended as medical advice.")
                print()

            # Automatically run evaluation if there's more than one drug
            if len(batch_summaries) > 1:
                print("\nEvaluating medication summaries...")
                evaluations = evaluator.evaluate_drug_summaries(batch_summaries)

                # Create visualization
                print("Generating visualization...")
                viz_file = visualize_drug_evaluations(evaluations, cond)
                if viz_file:
                    print(f"Evaluation visualization saved to {viz_file}")
                else:
                    print("No valid evaluation data available")

            print("\nWould you like to ask about another condition?\n")
        except Exception as e:
            print(f"Error processing your query: {e}")
            print("Please try again with different wording.")
            import traceback
            traceback.print_exc()  # Print stack trace for debugging


if __name__ == '__main__':
    main()
