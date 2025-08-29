# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 18:09:29 2025

@author: pathouli
"""

def jd_fun(str_a, str_b):
    tok_a = str_a.split()
    tok_b = str_b.split()
    set_a = set(tok_a)
    set_b = set(tok_b)
    inter_tmp = set_a.intersection(set_b)
    union_tmp = set_a.union(set_b)
    jd_tmp = len(inter_tmp) / len(union_tmp)
    return jd_tmp

def hello_world():
    print ("Hello World!")
    
def adder(a_in, b_in):
    tmp = None
    try:
        tmp = a_in + b_in
    except:
        print ("can't add", a_in, b_in)
        pass
    return tmp

def wrd_freq(c_in):
    tmp = c_in.split()
    t_d = dict()
    for word in set(tmp):
        t_d[word] = tmp.count(word)
    return t_d

def wrd_freq_redux(str_in):
    import collections
    the_new_ans = dict(collections.Counter(str_in.split()))
    return the_new_ans

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

def tok_cnt(c_in, sw_in):
    tmp = c_in.split()
    if sw_in == "all":
        return len(tmp)
    else:
        return len(set(tmp))

def wrd_fun(df_in):
    combined_t = dict()
    for topic in df_in["label"].unique():
        tmp = df_in[df_in["label"] == topic]
        tmp = tmp["body"].str.cat(sep= " ")
        tmp = tmp.split()
        tmp_all = len(tmp)
        tmp_u = len(set(tmp))
        combined_t[topic] = {"all": tmp_all, "unique": tmp_u}
        # combined = the_data.groupby('label', as_index=False).agg({'body': ' '.join})
        # combined['num_token'] = combined['body'].apply(lambda x: tok_cnt(x,"all"))
        # combined['num_token_unique'] = combined['body'].apply(lambda x: tok_cnt(x,"u"))
        # result_dict = {}
        # for _,row in combined.iterrows():
        #     result_dict[row['label']] = {
        #         'all': row['num_token'],
        #         'unique': row['num_token_unique']
        #     }
    return combined_t

def word_fun(df_in, c_name):
    import collections
    wrd_cnt_fun = dict()
    for topic in df_in["label"].unique():
        tmp = df_in[df_in["label"] == topic]
        tmp = tmp[c_name].str.cat(sep=" ")
        wrd_cnt_fun[topic] = collections.Counter(tmp.split())
    return wrd_cnt_fun
    
def rem_sw(text):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    return ' '.join([word for word in text.split() if word.lower() not in sw])

def stem_fun(input, sw_in):
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    tmp = input.split()
    if sw_in == "ps":
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in tmp]
    else:
        ps = WordNetLemmatizer()
        stemmed_words = [ps.lemmatize(
            word) for word in tmp]
    return ' '.join(stemmed_words)

def read_pickle(path_in, name_in):
    import pickle
    the_data_t = pickle.load(
        open(path_in + name_in + ".pk", "rb"))
    return the_data_t

def write_pickle(obj_in, path_in, name_in):
    import pickle
    pickle.dump(obj_in, open(
        path_in + name_in + ".pk", "wb"))

def vec_fun(df_in, col_n, m_in, n_in, p_o, sw_in):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import pandas as pd
    if sw_in == "tf":
        cv = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        cv = TfidfVectorizer(ngram_range=(m_in, n_in))
    xform_data_t = pd.DataFrame(cv.fit_transform(
        df_in[col_n]).toarray())
    xform_data_t.columns = cv.get_feature_names_out()
    write_pickle(cv, p_o, sw_in)
    return xform_data_t, cv

def cos_fun(df_a, df_b):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_fun = pd.DataFrame(cosine_similarity(df_a, df_b))
    cos_fun.index = df_a.index
    cos_fun.columns = df_a.index
    return cos_fun

def pca_fun(df_in, exp_v, p_o):
   from sklearn.decomposition import PCA
   import pandas as pd
   pca = PCA(n_components=exp_v)

   xform_data_dim = pd.DataFrame(pca.fit_transform(df_in))
   exp_var = sum(pca.explained_variance_ratio_)
   print ("Explained Var", exp_var)
   write_pickle(pca, p_o, "pca")
   return xform_data_dim

def clust_fun(df_in, t_l, n_c, o_p):
    num_clusters = n_c
    from sklearn.cluster import KMeans
    import pandas as pd
    kmeans_model = KMeans(
        n_clusters=num_clusters, random_state=42, n_init='auto',
        algorithm="elkan")
    kmeans_model.fit(df_in)
    cluster_assignments = pd.DataFrame(kmeans_model.labels_)
    cluster_assignments.index = t_l
    cluster_assignments.columns = ["cluster"]
    #cluster_assignments["truth"] = cluster_assignments.index.map(clus_dict)
    write_pickle(kmeans_model, o_p, "cluster")
    return cluster_assignments

def cluster_stats(c_in):
    c_dict = dict()
    for c in set(c_in.index):
        tmp = c_in[c_in.index == c]
        tmp = tmp.groupby("cluster").agg(total=("cluster", "count"))
        c_dict[c] = tmp
    
    clust_dict = dict()
    for k in c_dict.keys():
        tmp = c_dict[k].idxmax()
        clust_dict[k] = tmp[0]
    return clust_dict

def extract_embeddings_pre(df_in, out_path_i, name_in):
    #https://code.google.com/archive/p/word2vec/
    #https://pypi.org/project/gensim/
    #pip install gensim
    #name_in = 'models/word2vec_sample/pruned.word2vec.txt'
    import pandas as pd
    from nltk.data import find
    from gensim.models import KeyedVectors
    import pickle
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(my_model_t.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    word2vec_sample = str(find(name_in))
    my_model_t = KeyedVectors.load_word2vec_format(
        word2vec_sample, binary=False)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model_t, open(out_path_i + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(out_path_i + "embeddings_df.pkl", "wb" ))
    return tmp_data, my_model_t

def domain_train(df_in, path_in, name_in):
    #domain specific
    import pandas as pd
    import gensim
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(model.wv.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    model = gensim.models.Word2Vec(df_in.str.split())
    model.save(path_in + 'body.embedding')
    #call up the model
    #load_model = gensim.models.Word2Vec.load('body.embedding')
    #model.wv.similarity('fish','river')
    tmp_data = pd.DataFrame(df_in.str.split().apply(get_score))
    tmp_data = tmp_data.apply(lambda x: {
        f"{k}": vv for v in x for k, vv in enumerate(v, 0)
        },result_type="expand", axis=1,
)
    return tmp_data, model

def llm_fun(df_in, p_in, n_in):
    #https://pypi.org/project/sentence-transformers/
    #https://huggingface.co/models?library=sentence-transformers
    #pip install --upgrade "optree>=0.13.0"
    #pip install -U sentence-transformers
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    if n_in == "small":
        llm = 'sentence-transformers/all-MiniLM-L6-v2'
    else:
        llm = 'sentence-transformers/all-mpnet-base-v2' #takes time but superb performance
    model = SentenceTransformer(llm)
    write_pickle(model, p_in, n_in)
    vec_t = pd.DataFrame(model.encode(df_in))
    return vec_t

def chi_fun(df_in, lab_in, k_in, p_in, n_in, stat_sig):
    from sklearn.feature_selection import chi2, SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=k_in)
    dim_data = pd.DataFrame(feat_sel.fit_transform(df_in, lab_in))
    p_val = pd.DataFrame(list(feat_sel.pvalues_))
    p_val.columns = ["pval"]
    feat_index = list(p_val[p_val.pval <= stat_sig].index)
    dim_data = dim_data[feat_index]
    feature_names = df_in.columns[feat_index]
    dim_data.columns = feature_names
    write_pickle(feat_sel, p_in, n_in)
    write_pickle(dim_data, p_in, "chi_data_" + n_in)
    return dim_data, feat_sel

def model_fun(df_in, lab_in, sw_i, t_s, o_p):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd
    if sw_i == "rf":
        model = RandomForestClassifier(
            random_state=123)
    elif sw_i == "gnb":
        model = GaussianNB()
    elif sw_i == "gbc":
        model = GradientBoostingClassifier()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, lab_in, test_size=t_s,
        random_state=42)
    
    model.fit(X_train, y_train)
    write_pickle(model, o_p, sw_i)
    y_pred = model.predict(X_test)
    y_pred_prob = pd.DataFrame(model.predict_proba(X_test))
    y_pred_prob.columns = model.classes_
    
    the_metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    
    the_metrics.index = ["precision", "recall", "fscore", None]
    print (the_metrics)
    try:
        feat_imp = pd.DataFrame(model.feature_importances_)
        feat_imp.index = df_in.columns
        feat_imp.columns = ["score"]
        perc_prop = feat_imp[feat_imp["score"] > 0]
        percent = len(perc_prop)/len(feat_imp)*100
        print ("Percent with propensity", percent, "%")
        feat_imp.to_csv(o_p + sw_i + ".csv")
    except:
        print (sw_i, "does not support FI")
        pass
    return model

def model_grid_fun(df_in, lab_in, sw_i, t_s, c_v, g_in, o_p):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    import pandas as pd
    if sw_i == "rf":
        model = RandomForestClassifier(
            random_state=123)
    elif sw_i == "gnb":
        model = GaussianNB()
    elif sw_i == "gbc":
        model = GradientBoostingClassifier()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, lab_in, test_size=t_s,
        random_state=42)
    
    g_s = GridSearchCV(model, g_in, cv=c_v)
    g_s.fit(X_train, y_train)
    
    best_params = g_s.best_params_
    print ("Best params", best_params)
    best_perf = g_s.best_score_
    print ("Best perf", best_perf)
    
    if sw_i == "rf":
        model = RandomForestClassifier(**g_s.best_params_,
            random_state=123)
    elif sw_i == "gnb":
        model = GaussianNB(**g_s.best_params_)
    elif sw_i == "gbc":
        model = GradientBoostingClassifier(**g_s.best_params_)
    
    model.fit(df_in, lab_in)
    write_pickle(model, o_p, sw_i)
    
    try:
        feat_imp = pd.DataFrame(model.feature_importances_)
        feat_imp.index = df_in.columns
        feat_imp.columns = ["score"]
        perc_prop = feat_imp[feat_imp["score"] > 0]
        percent = len(perc_prop)/len(feat_imp)*100
        print ("Percent with propensity", percent, "%")
        feat_imp.to_csv(o_p + sw_i + ".csv")
    except:
        print (sw_i, "does not support FI")
        pass
    return model

# def stem_fun(input):
#     from nltk.stem import PorterStemmer 
#     ps = PorterStemmer()
#     tmp = input.split()
#     stemmed_words = [ps.stem(word) for word in tmp]
#     return ' '.join(stemmed_words)

# def rem_sw(str_in):
#     n_l = list()
#     from nltk.corpus import stopwords
#     sw = stopwords.words('english')
#     # n_l = [word for word in t_c.split() if word not in sw]
#     # n_l = " ".join(n_l)
#     for word in str_in.split():
#         if word not in sw:
#             n_l.append(word)
#     n_l = " ".join(n_l)
#     return n_l

# def file_opener(p_in):
#     try:
#         txt = ''
#         #txt = None
#         f = open(p_in, "r", encoding="UTF8")
#         txt = f.read()
#         txt = clean_text(txt)
#         f.close()
#     except:
#         print ("Can't open", p_in)
#         pass
#     return txt