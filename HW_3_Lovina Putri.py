# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:31:01 2025

@author: lovin
"""
NAME = "Lovina Putri" 
UNI = "lap2236"   

# Import required packages and functions
from utils import *
import nltk
import os
import pandas as pd
nltk.download('stopwords')  

# Set directories
the_d_path = "C:/Users/lovin/nlp/hw3/data/"
the_path = "C:/Users/lovin/nlp/hw3/output/"

# Load and clean the corpus into a DataFrame with `body` column
the_data = file_crawler(the_d_path)

# Remove stopwords from 'body' and creates 'body_sw'
the_data["body_sw"] = the_data["body"].apply(rem_sw)

# Stem the stopword-removed text → creates 'body_sw_stem'
the_data["body_sw_stem"] = the_data["body_sw"].apply(lambda x: stem_fun(x, "ps"))

# Save the data result 
write_pickle(the_data, the_path, "the_data")

# Function
def word_prob(token_seq: str, col: str, df: pd.DataFrame, output_dir: str = None, save_name: str = None) -> dict:
    """
    Calculates the probability that a token/phrase appears in each topic corpus.

    Parameters:
    - token_seq (str): Token or phrase to search (e.g. 'machine learning')
    - col (str): Column to use ('body', 'body_sw', 'body_sw_stem')
    - df (pd.DataFrame): DataFrame containing the text data and 'label'
    - output_dir (str): Optional directory to save results as CSV
    - save_name (str): Optional file name for CSV (e.g., 'output.csv')

    Returns:
    - Dictionary with topic -> probability (or None if not found)
    """

    # Validation
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the dataframe.")
    if not isinstance(token_seq, str) or not token_seq.strip():
        raise ValueError("token_seq must be a non-empty string.")
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column for topic classification.")

    topics = ['all', 'fishing', 'hiking', 'machinelearning', 'mathematics']
    prob_dict = {}

    for topic in topics:
        sub_df = df if topic == 'all' else df[df['label'] == topic]
        text_series = sub_df[col].dropna()
        all_text = " ".join(text_series)
        total_tokens = len(all_text.split())
        count = all_text.count(token_seq)
        prob_dict[topic] = None if count == 0 or total_tokens == 0 else count / total_tokens

    # Save to CSV 
    if output_dir and save_name:
        os.makedirs(output_dir, exist_ok=True)
        out_df = pd.DataFrame([prob_dict])
        out_df.insert(0, "token", token_seq)
        save_path = os.path.join(output_dir, save_name)
        out_df.to_csv(save_path, index=False)
        print(f"Saved to: {save_path}")

    return prob_dict

# Test block
if __name__ == "__main__":
    # Sample token
    test_token = "data"
    # Sample column (body_sw_stem)
    test_col = "body_sw_stem"
    
    # Succeed test
    try:
        print("Running valid test case...")
        output = word_prob(test_token, test_col, the_data, output_dir=the_path, save_name="test_output.csv")
        print("Output:", output)
    except Exception as e:
        print("Test failed:", e)

    # Invalid column
    try:
        print("\nRunning invalid column test...")
        word_prob(test_token, "invalid_column", the_data)
    except ValueError as e:
        print("Expected error:", e)

    # Invalid token
    try:
        print("\nRunning invalid token test...")
        word_prob("", test_col, the_data)
    except ValueError as e:
        print("Expected error:", e)