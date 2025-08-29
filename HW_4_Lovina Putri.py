#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 17:16:56 2025

@author: lovin
"""
NAME = "Lovina Putri"
UNI  = "...."

import os
import praw
import pandas as pd
import pytz
from datetime import datetime
from utils import *
import warnings
warnings.filterwarnings("ignore")

# === Directory ===
the_path    = "[path]/hw4/data/"
output_path = "[path]/hw4/output/"
os.makedirs(output_path, exist_ok=True)

subreddit_channel = 'MachineLearning+fishing'

reddit = praw.Reddit(
    client_id="[your client id]",
    client_secret="[your client secret]",
    user_agent="testscript by u/fakebot3",
    username="[your username]",
    password="[your password]",
    check_for_async=False
)
print(reddit.read_only)

# ---------- The helpers ----------
def conv_time(var):
    tmp_df = pd.DataFrame().append({'created_at': var}, ignore_index=True)
    tmp_df.created_at = pd.to_datetime(tmp_df.created_at, unit='s')\
                           .dt.tz_localize('utc')\
                           .dt.tz_convert('US/Eastern')
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    # return a plain dict { 'body': ... }
    try:
        return {'body': var_in.body}
    except Exception as e:
        print("ERROR:", e)
        return {}

# ---------- Training + inference ----------
# clean_text -> rem_sw -> stem_fun("ps") -> vec_fun(tf,1,1) -> pca_fun(0.95) -> RF(grid)

def preprocess_text(t: str) -> str:
    t = clean_text(t)
    t = rem_sw(t)
    t = stem_fun(t, "ps")
    return t

def artifacts_exist():
    return (
        os.path.exists(os.path.join(output_path, "tf.pk")) and
        os.path.exists(os.path.join(output_path, "pca.pk")) and
        os.path.exists(os.path.join(output_path, "rf.pk"))
    )

def train_pipeline():
    """
    Train ONLY on the 'fishing' and 'machinelearning' corpora in the_path
    and save artifacts into output_path: tf.pk, pca.pk, rf.pk
    """
    print(">> Training pipeline starting...")
    # 1) Load data
    the_data = file_crawler(the_path)
    the_data = the_data[the_data['label'].isin(['fishing', 'machinelearning'])].reset_index(drop=True)

    # 2) Preprocess
    the_data['body'] = the_data['body'].apply(clean_text)
    the_data['body_sw'] = the_data['body'].apply(rem_sw)
    the_data['body_sw_stem'] = the_data['body_sw'].apply(lambda x: stem_fun(x, 'ps'))

    # 3) Vectorize (tf, 1-gram) -> saves tf.pk
    X, vec = vec_fun(the_data, 'body_sw_stem', 1, 1, output_path, 'tf')

    # 4) PCA (0.95) -> saves pca.pk
    Xp = pca_fun(X, 0.95, output_path)

    # 5) RF + grid search -> saves rf.pk
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 20, 50],
        'max_features': ['sqrt', 'log2', None]
    }
    y = the_data.label

    try:
        from utils import model_grid_fun
        model = model_grid_fun(Xp, y, 'rf', 0.2, 5, param_grid, output_path)
    except Exception as e:
        print("model_grid_fun not available; falling back to sklearn GridSearchCV. Reason:", e)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV, train_test_split
        import pickle
        X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.2, random_state=123, stratify=y)
        grid = GridSearchCV(
            RandomForestClassifier(random_state=123, n_jobs=-1),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        try:
            write_pickle(output_path, "rf", best)
        except Exception:
            with open(os.path.join(output_path, "rf.pk"), "wb") as f:
                pickle.dump(best, f)
        model = best

    print("Training complete. Artifacts saved to:", output_path)
    print("   - Vectorizer: tf.pk")
    print("   - PCA:        pca.pk")
    print("   - Model:      rf.pk")
    return model

def load_artifacts():
    vec = read_pickle(output_path, "tf")
    pca = read_pickle(output_path, "pca")
    model = read_pickle(output_path, "rf")
    return vec, pca, model

def predict_one(text, vec=None, pca=None, model=None):
    """Predict class + probability for a single text string."""
    import pandas as pd
    if vec is None or pca is None or model is None:
        vec, pca, model = load_artifacts()
    t_prep = preprocess_text(text)
    X  = pd.DataFrame(vec.transform([t_prep]).toarray())
    Xp = pd.DataFrame(pca.transform(X))
    pred  = model.predict(Xp)[0]
    proba = model.predict_proba(Xp)[0]
    classes = list(model.classes_)
    try:
        score = float(proba[classes.index(pred)])
    except Exception:
        score = float(max(proba))
    all_probs = {cls: float(p) for cls, p in zip(classes, proba)}
    return pred, score, all_probs

# ---------- Ensure artifacts (train if missing), then load once ----------
if not artifacts_exist():
    train_pipeline()
vec, pca, model = load_artifacts()

# ---------- Additional: quick test (set True to test and exit) ----------
RUN_QUICK_TEST = False
TEST_SAMPLES = [
    "Caught a big bass with a spinnerbait near the pier.",
    "We fine-tuned a transformer with LoRA on a small dataset."
]
if RUN_QUICK_TEST:
    print("\n>>> QUICK TEST")
    for i, txt in enumerate(TEST_SAMPLES, 1):
        label, score, all_probs = predict_one(txt, vec=vec, pca=pca, model=model)
        preview = txt if len(txt) <= 240 else (txt[:240] + "...")
        print("\n" + "—"*40)
        print(f"#{i} BODY:", preview)
        print(f"PRED: {label} | PROB: {score:.3f} | ALL_PROBS: {all_probs}")
    import sys
    sys.exit(0)

# ---------- Final: stream + classify ----------
for comment in reddit.subreddit(subreddit_channel).stream.comments(skip_existing=True):
    tmp = get_reddit_data(comment)
    body = tmp.get("body", "")
    if not body:
        continue

    label, score, all_probs = predict_one(body, vec=vec, pca=pca, model=model)

    preview = body.replace("\n", " ").strip()
    if len(preview) > 240:
        preview = preview[:240] + "..."

    print("\n" + "—"*40)
    print("BODY:", preview)
    print(f"PRED: {label} | PROB: {score:.3f} | ALL_PROBS: {all_probs}")


