# Natural Language Programming

This repo contains small NLP assignments and a mini-project written in pure Python. It covers text cleaning, token statistics, feature extraction (TF/TF-IDF), similarity, dimensionality reduction, clustering, and a live Reddit classifier.

> Data: the mini-project trains on a tiny local corpus (two folders: `data/machinelearning/`, `data/fishing/`) and then classifies live Reddit comments from those subreddits. No raw data is redistributed.

## Outputs (scripts)
* HW_1_Lovina Putri.py — warm-up utilities & string ops (token/Jaccard counts, word frequencies).
* HW_2_Lovina Putri.py — feature engineering basics and similarity experiments.
* HW_3_Lovina Putri.py — vectorization, PCA, and simple clustering.
* HW_4_Lovina Putri.py — end-to-end pipeline: clean → remove stopwords → stem/lemmatize → vectorize (CountVectorizer) → PCA (0.95 VE) → Random Forest (grid search), plus a Reddit stream classifier.
* utils.py — helper functions shared across scripts (e.g., clean_text, rem_sw, stem_fun, vec_fun, pca_fun, clust_fun, model/pickle I/O). [created by the instructor]

## Notes
* Requires Python 3.10+.
* Core libs: `nltk`, `scikit-learn`, `praw`. Optional: `gensim`, `sentence-transformers`.

## Author
Built by Lovina Aisha Malika Putri as coursework/mini-projects in NLP. Feedback and PRs welcome.
