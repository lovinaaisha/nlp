# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 22:55:39 2025

@author: lovina
"""
NAME = "Lovina Putri" 
UNI = "...."   

import os
import re
import pandas as pd
import math

data_path = "[path]/hw2/data/"
output_path = "[path]/hw2/output/"
os.makedirs(output_path, exist_ok=True)

# =============================================================================
#                                 Question 1
# =============================================================================
def load_lexicon(fp: str) -> set[str]:
    lex = set()
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith(";"):
                lex.add(w.lower())
    return lex

pw = load_lexicon(os.path.join(data_path, "positive-words.txt"))
nw = load_lexicon(os.path.join(data_path, "negative-words.txt"))

def gen_senti(text: str,
              pos_set: set[str],
              neg_set: set[str]
             ) -> float | None:
    tokens = re.findall(r"\b[\w']+\b", text.lower())
    scores = [1 if t in pos_set else -1
              for t in tokens
              if t in pos_set or t in neg_set]
    if not scores:
        return None
    return sum(scores) / len(scores)

def test_simple_senti():
    """Test Question 2: 'simple_senti' column via gen_senti without pytest."""
    print("Testing simple_senti…")

    

    df = pd.DataFrame({
        "body": [
            "good good good",    # all pos → +1
            "bad bad",           # all neg → -1
            "good bad good",     # mix → (1-1+1)/3 = 1/3
            "no sentiment here"  # none → NaN
        ]
    })

    df["simple_senti"] = df["body"].apply(lambda txt: gen_senti(txt, pw, nw))

    expected = [1.0, -1.0, 1/3, None]
    tol = 1e-8

    for i, exp in enumerate(expected):
        actual = df.loc[i, "simple_senti"]
        if exp is None:
            assert actual is None or (isinstance(actual, float) and math.isnan(actual)), \
                f"❌ simple_senti Test {i+1} Failed: expected None/NaN, got {actual!r}"
        else:
            assert actual is not None and abs(actual - exp) < tol, \
                f"❌ simple_senti Test {i+1} Failed: expected {exp}, got {actual}"
    print("✅ simple_senti: All tests passed!")

# Run the test

def run_all_tests():
    # … your existing Q-tests …
    test_simple_senti()
    print("="*50)
    print("All HW 2 tests passed.")
    print("="*50)


if __name__ == "__main__":
    # … header prints …
    run_all_tests()

# =============================================================================
#                                 Question 2
# =============================================================================
def clean_text(str_in):
    import re
    cln_txt = re.sub(
        "[^A-Za-z']+", " ", str_in).strip().lower()
    return cln_txt

def file_opener(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return clean_text(f.read())
    except:
        print ("Can't open", file_path)
        return ''
        pass
    
def file_crawler(p_in):
    import pandas as pd
    import os
    tmp_d = pd.DataFrame()
    for root, dirs, files in os.walk(p_in, topdown=False):
       for name in files:
          tmp = root + "/" + name
          t_txt = file_opener(tmp)
          if len(t_txt) != 0:
          #if t_txt is not None: 
              t_dir = root.split("/")[-1]
              tmp_pd = pd.DataFrame(
                  {"body": t_txt, "label": t_dir}, index=[0])
              tmp_d = pd.concat([tmp_d, tmp_pd], ignore_index=True)
    return tmp_d

the_data = file_crawler(data_path)
the_data["simple_senti"] = the_data["body"].apply(lambda txt: gen_senti(txt, pw, nw))

# =============================================================================
#                                 Question 3
# =============================================================================
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
the_data["vader"] = the_data["body"].apply(
    lambda txt: analyzer.polarity_scores(txt)["compound"]
)

# =============================================================================
#                                 Question 4
# =============================================================================
summary = the_data[["simple_senti", "vader"]].agg(["mean", "median", "std"])

print(summary)
