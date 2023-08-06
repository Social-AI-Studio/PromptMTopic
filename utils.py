from itertools import chain
from collections import Counter
import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import pickle
from topic_metrics.counting import split_corpus, count_histogram, count_vocab, count_windows
from topic_metrics.measuring import single_count_setup, create_joint_prob_graph, create_graph_with, npmi
from topic_metrics.measuring import calculate_scores_from_counts, direct_avg
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
import os

stop_words=stopwords.words('english')
stop_words=[token.lower() for token in stop_words]
punctuation = "!\"#$%&'()*+,./:;<=>?@[\]^_`{|}~"

def get_most_freq_words(caption_file, size=30):    
    with open(caption_file,"r") as f:
        out=json.load(f)
    docs = [out[x] for x in out]
    docs = [re.sub(r'[^\w\s]', '', doc) for doc in docs]
    all_words_2d = [doc.split() for doc in docs]
    all_words = list(chain.from_iterable(all_words_2d))
    all_words = [word.lower() for word in all_words if word.lower() not in stop_words]    
    counts = Counter(all_words)
    most_freq_words = []
    for k, v in counts.most_common(size):
        most_freq_words.append(k)        
    return most_freq_words

def documents_preprocess(docs,lemmatize=True):
    docs = [str(doc).lower() for doc in docs]
    # Remove username
    docs = [re.sub(r"@[a-zA-Z0-9_]+", '', str(doc)) for doc in docs]
    # Remove url
    docs = [re.sub(r'(?i)http\S+', '', str(doc)) for doc in docs] 
    docs = [re.sub(r'(?i)www\S+', '', str(doc)) for doc in docs]
    web = set()
    for doc in docs:
        match = re.search(r"(?i)\b\w+(?=\.com|\.net|\.org|\.xyz|\.in|\.co|\.con|\.eu|\.us|\.es)", str(doc))
        if match:
            web.add(match.group())
    docs = [re.sub(r"(?i)\b\S+(?:\.com|\.net|\.org|\.xyz|\.in|\.co|\.con|\.eu|\.us|\.es)\b",'', str(doc)) for doc in docs]
    # Remove punctuation
    docs = [doc.translate(str.maketrans(' ',' ',punctuation)) for doc in docs]  
    docs = [doc.strip().replace("  ", " ").replace("  ", " ") for doc in docs]
    
    docs_list = [sent.split(" ") for sent in docs]
    
    # Lemmatization + Remove stop words
#     filtered_docs =[[stemmer.stem(w) for w in sent] for sent in docs_list]
    filtered_docs = [[w for w in sent if not w in stop_words] for sent in docs_list]
    filtered_docs = [[w for w in sent if not w in web] for sent in filtered_docs]
    filtered_docs = [" ".join(sent) for sent in filtered_docs]    
    return filtered_docs

def calc_metrics(dataset,input_folder,output_folder,k):
    corpus_dir = os.path.join(input_folder,dataset, 'corpus')
    dest_dir="{}/{}".format(output_folder,dataset)
    os.makedirs(corpus_dir, exist_ok=True)
    count_vocab(corpus_dir, dest_dir, num_processes=4)

    # # # # # uploaded vocab are in alphabetically-sorted
    vocab = sorted(pickle.load(open(f"{dest_dir}/vocab_count.pkl", 'rb')))
    vocab_index = {k:i for i,k in enumerate(vocab)}
    count_histogram(corpus_dir,dest_dir, num_processes=4)
    count_windows(corpus_dir, dest_dir, window_size=10, vocab2id=vocab_index) 

    dest_npmi = '{}/{}/20ng/npmi_10'.format(output_folder,dataset)
    os.makedirs(dest_npmi, exist_ok=True)

    num_windows, single_prob = single_count_setup("{}/{}/histogram.csv".format(output_folder,dataset),
                                                "{}/{}/10/single.pkl".format(output_folder,dataset),
                                                window_size=10, min_freq=0)

    joint_prob = create_joint_prob_graph("{}/{}/10/joint".format(output_folder,dataset), 
                            num_windows=num_windows, min_freq=0, shortlist=[],
                            num_processes=20, existing_graph={})

    """
    Using the probability graph, we can create scored (npmi) graphs
    """
    npmi_graph = create_graph_with(npmi, joint_prob, single_prob, smooth=True)

    """
    Saving it in a similar format as count graphs
    """
    for k1, s in npmi_graph.items():
        pickle.dump(dict(s), open(os.path.join(dest_npmi,f"{k1}.pkl"), "wb"))
    
    with open("{}/{}/topic_rep.json".format(output_folder,dataset),"r") as f:
        topics=json.load(f)
    topics=[words for _,words in topics.items()]    
    scores = calculate_scores_from_counts([[vocab_index[w] for w in t] for t in topics], 
                    "{}/{}/histogram.csv".format(output_folder,dataset),
                    "{}/{}/10/single.pkl".format(output_folder,dataset),
                    "{}/{}/10/joint".format(output_folder,dataset), 
                    score_func = npmi, window_size = 10, agg_func = direct_avg,
                    smooth=True, min_freq=0, num_processes=10)

    metric=0
    for topic, score in zip(topics, scores):
        metric+=score
    print("dataset:{} k:{} npmi: {}".format(dataset,k,metric/k))    

    topic_tokens={'topics':topics}   
    diversity = TopicDiversity(topk=10).score(topic_tokens) 
    print("dataset:{} k:{} diversity: {}".format(dataset,k,diversity))