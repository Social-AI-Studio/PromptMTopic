import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# from text_preprocessing import generate_filtered_docs
from utils import *

# Return the lists of topic names
# Soft: whether allow one image to be assigned to multiple topics 
def return_first_topic(doc_path, soft=False):
    f = open(doc_path, "r")
    documents_topics = list(f.readlines())
    
    topic_set = []
    for i in range(len(documents_topics)):
        record = documents_topics[i]
        split_index = record.find('''': "[''')
        if(split_index > 0):
            end_idx = record.find("]")
            topic_label = [word.strip().lower().replace("'", "") for word in record[split_index+5 : end_idx].split(",")]
            if(soft):
                for topic in topic_label:
                    if(len(topic.split()) <= 5):
                        topic_set.append(topic)
                    else:
                        topic_set.append("Miscellaneous")
            else:
                if(len(topic_label[0].split()) <= 5):
                    topic_set.append(topic_label[0])
                else:
                    topic_set.append("Miscellaneous")

        else:
            topic_set.append("Miscellaneous")
    return list(set(topic_set))


# Return the grouping of documents and images based on topic they assigned to before collapsing
def retrieve_topic_doc(doc_path, map_path, cap_txt, soft=False):
    
    f = open(doc_path, "r")
    documents_topics = list(f.readlines())
    
    if(map_path != None):
        with open(map_path, "r") as f:
            mapping = json.load(f)
            o_c_mapping = {}
            for key in mapping.keys():
                o_c_mapping[mapping[key]] = key

    topic_set = return_first_topic(doc_path, soft)
    topics_documents = {}
    image_documents = {}
    for topic in topic_set:
        topics_documents[topic] = []
        image_documents[topic] = []

    for i in range(len(documents_topics)):
        record = documents_topics[i]
        split_index = record.find('''': "[''')
        if(split_index > 0):
            end_idx = record.find("]")
            topic_label = [word.strip().lower().replace("'", "") for word in record[split_index+5 : end_idx].split(",")]
            img_label = record[2:split_index]
            
            if(map_path != None):
                if(o_c_mapping[img_label] in cap_txt.keys()):
                    caption_text = cap_txt[o_c_mapping[img_label]] 
                   
                    if(soft):
                        for topic in topic_label:
                            if(len(topic.split()) > 5):
                                topic = "Miscellaneous"
                            image_documents[topic].append(img_label)
                            topics_documents[topic].append(caption_text)
                    else:
                        if(len(topic_label[0].split()) > 5):
                            topic_label[0] = "Miscellaneous"
                        image_documents[topic_label[0]].append(img_label)
                        topics_documents[topic_label[0]].append(caption_text)
                else:
                    continue
            else:
                if(img_label in cap_txt.keys()):
                    caption_text = cap_txt[img_label]
                    
                    if(soft):
                        for topic in topic_label:
                            if(len(topic.split()) > 5):
                                topic = "Miscellaneous"
                            image_documents[topic].append(img_label)
                            topics_documents[topic].append(caption_text)
                    else:
                        if(len(topic_label[0].split()) > 5):
                            topic_label[0] = "Miscellaneous"
                        image_documents[topic_label[0]].append(img_label)
                        topics_documents[topic_label[0]].append(caption_text)
                else:
                    continue
            
        else:
            split_index = record.find('''': ''')
            img_label = record[2:split_index]
            
            if(map_path != None):
                if(o_c_mapping[img_label] in cap_txt.keys()):
                    caption_text = cap_txt[o_c_mapping[img_label]] 
                    image_documents["Miscellaneous"].append(img_label)
                    topics_documents["Miscellaneous"].append(caption_text)
                else:
                    continue
            else:
                if(img_label in cap_txt.keys()):
                    caption_text = cap_txt[img_label]
                    image_documents["Miscellaneous"].append(img_label)
                    topics_documents["Miscellaneous"].append(caption_text)
                else:
                    continue
            
    return topics_documents, image_documents


# +
# Return the similarity score between two top words lists
def calculate_overlap(list1, list2):
    overlap = 0.0
    for word in list1:
        if(word in list2):
            overlap += 1.0
    return overlap / min(len(list1), len(list2))

# Return the top 20 words for each topic
def get_top20_words_dic(topics_documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform([" ".join(topics_documents[key]) for key in topics_documents.keys()]) 
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    
    top_words_dic = {}
    
    for i in range(len(topics_documents.keys())):
        key = list(topics_documents.keys())[i]
        first_document_vector = tfidf[i] 

        feature_array = np.array(tfidf_feature_names)
        tfidf_sorting = np.argsort(first_document_vector.toarray()).flatten()[::-1]
        
        non_zero_list = np.count_nonzero(first_document_vector.toarray())
        top20 = 20
        if(non_zero_list < top20):
            top20 = non_zero_list
            
        top_n = feature_array[tfidf_sorting][:top20]
        top_words_dic[key] = top_n
        
    return top_words_dic
    
# Convert similarity dictionary to matrix
def convert_dic_to_matrix(matrix):
    topic_list = list(matrix.keys())
    value_matrix = [[] for i in range(len(topic_list))]
    for i in range(len(topic_list)):
        topic = topic_list[i]
        for key in topic_list:
            if(key not in matrix[topic].keys() and key == topic):
                value_matrix[i].append(-1)
            else:
                value_matrix[i].append(matrix[topic][key])
    return topic_list, value_matrix


# -

# Calculate word similarity between all topics
def calculate_word_similarity_matrix(topics_documents):
    word_overlapping_matrix = {}
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform([" ".join(topics_documents[key]) for key in topics_documents.keys()]) 
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    
    top_words_dic = {}
    
    for i in range(len(topics_documents.keys())):
        key = list(topics_documents.keys())[i]
        first_document_vector = tfidf[i] 

        feature_array = np.array(tfidf_feature_names)
        tfidf_sorting = np.argsort(first_document_vector.toarray()).flatten()[::-1]
        
        non_zero_list = np.count_nonzero(first_document_vector.toarray())
        top20 = 20
        if(non_zero_list < top20):
            top20 = non_zero_list
            
        top_n = feature_array[tfidf_sorting][:top20]
        top_words_dic[key] = top_n
        
    
    for i in range(len(topics_documents.keys())):
        key1 = list(topics_documents.keys())[i]
        list1 = top_words_dic[key1]
        word_overlapping_matrix[key1] = {}
        
        for j in range(len(topics_documents.keys())):
            if(j!=i):
                key2 = list(topics_documents.keys())[j]
                list2 = top_words_dic[key2]
                score = calculate_overlap(list1, list2) 
                word_overlapping_matrix[key1][key2] = score 
    return top_words_dic, word_overlapping_matrix

# + 0.5 / (1+ np.log(len( topics_documents[key1])))

# -

#  Collapse documents to predefined number of topics
def WSM_collapsing(topics_documents, k, word_overlapping_matrix, topic_grouping = None):
    top_words_dic = get_top20_words_dic(topics_documents)
    topic_dic, value_matrix = convert_dic_to_matrix(word_overlapping_matrix)
    a = np.array(value_matrix)
    if(topic_grouping == None):
        topic_grouping = {}
    #     for topic in topic_dic:
    #         topic_grouping[topic] = ["emp"]
    
    while(a.shape[0] > k):
        if(a.shape[0] %10 == 0):
            print(a.shape)
        max_idx = np.unravel_index(a.argmax(), a.shape)
        max_row, max_col = max_idx
        if(a[max_row][max_col] <= 0):
            print("No overlapping")
            print(a.shape)
        sum_row = sum(a[max_row])
        sum_col = sum(a[max_col])
        
        del_idx = 0
        if(sum_col > sum_row):
            del_idx = max_row
            kep_idx = max_col
        else:
            del_idx = max_col
            kep_idx = max_row
        
        kep_topic = topic_dic[kep_idx]
        del_topic = topic_dic[del_idx]
        
        topics_documents[kep_topic] += topics_documents[del_topic]
        # image_documents[kep_topic] += image_documents[del_topic]
        del topics_documents[del_topic]
        # del image_documents[del_topic]
        
        
        additional_topic = []
        if(del_topic in topic_grouping and "del" not in topic_grouping[del_topic]):
            additional_topic = topic_grouping[del_topic]
        topic_grouping[del_topic] = ["del"]
        if(kep_topic not in topic_grouping):
            topic_grouping[kep_topic] = [del_topic] + additional_topic
        else:
            topic_grouping[kep_topic] += [del_topic] + additional_topic
            
        i = list(topics_documents.keys()).index(kep_topic)
        tfidf_vectorizer = TfidfVectorizer()
        tfidf = tfidf_vectorizer.fit_transform([" ".join(topics_documents[key]) for key in topics_documents.keys()]) 
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        first_document_vector = tfidf[i] 
        feature_array = np.array(tfidf_feature_names)
        tfidf_sorting = np.argsort(first_document_vector.toarray()).flatten()[::-1]
        non_zero_list = np.count_nonzero(first_document_vector.toarray())
        top20 = 20
        if(non_zero_list < top20):
            top20 = non_zero_list
        top_n = feature_array[tfidf_sorting][:top20]
        del top_words_dic[del_topic]
        top_words_dic[kep_topic] = top_n
        
        
        del word_overlapping_matrix[del_topic]
        word_overlapping_matrix[kep_topic] = {}
        for key in word_overlapping_matrix:
            if(key != kep_topic):
                del word_overlapping_matrix[key][del_topic]
                list1 = top_words_dic[kep_topic]
                list2 = top_words_dic[key]
                word_overlapping_matrix[kep_topic][key] = calculate_overlap(list1, list2)  
                
                list1 = top_words_dic[key]
                list2 = top_words_dic[kep_topic]
                word_overlapping_matrix[key][kep_topic] = calculate_overlap(list1, list2)  

        topic_dic, value_matrix = convert_dic_to_matrix(word_overlapping_matrix)
        a = np.array(value_matrix)
    # return topic_grouping, topics_documents, image_documents, word_overlapping_matrix
    return topics_documents
