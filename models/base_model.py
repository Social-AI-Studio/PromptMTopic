import os
from ..utils import *
from ..ctfidf import *
from sklearn.feature_extraction.text import CountVectorizer
from ..WSM import *

class Base_Model():
    def __init__(self) -> None:
        pass
    
    def collect_topic_rep(self,topic_docs,dataset,input_folder):
        with open(os.path.join(input_folder,dataset,"captions.json"),"r") as f:
            file_captions=json.load(f)
        with open(os.path.join(input_folder,dataset,"text.json"),"r") as f:
            file_text=json.load(f)

        caption_stopwords = get_most_freq_words(os.path.join(input_folder,dataset,"captions.json"))
        def remove_stop_rords(text):
            return [token.lower() for token in text.split() if token.lower() not in caption_stopwords]
        class_docs=[]
        for topic,docs in topic_docs.items():
            documents =[[token for token in remove_stop_rords(file_captions[file]) if token not in stop_words]+[token for token in file_text[file].split()] for file in docs]
            documents=[" ".join([token for token in doc if token not in stop_words]) for doc in documents]
            class_docs.append(" ".join(documents))  
            
        count_vectorizer=CountVectorizer(stop_words='english')
        tf = count_vectorizer.fit_transform(class_docs) 
        words = count_vectorizer.get_feature_names()
        vectorizer_model=ClassTfidfTransformer()
        transformer = vectorizer_model.fit(tf)
        tfidf = transformer.transform(tf)
        token_words=vectorizer_model._extract_words_per_topic(words,topic_docs,tfidf,100)
        return token_words
    
    def group_topics(self,topic_docs,dataset,input_folder,output_folder,k_range):
        with open(os.path.join(input_folder,dataset,"captions.json"),"r") as f:
            captions=json.load(f)
        with open(os.path.join(input_folder,dataset,"text.json"),"r") as f:
            text=json.load(f)

        caption_stopwords = get_most_freq_words(os.path.join(input_folder,dataset,"captions.json"))
        def remove_caption_stop_words(text):
            return [token.lower() for token in text.split() if token.lower() not in caption_stopwords]
        topic_docs={topic:["{} {}".format(text[doc].lower(),remove_caption_stop_words(captions[doc].lower())) for doc in docs] for topic,docs in topic_docs.items()}
        topic_docs={topic:documents_preprocess(docs,lemmatize=False) for topic,docs in topic_docs.items()}
        
        _, word_similarity_matrix = calculate_word_similarity_matrix(topic_docs)
        for k in k_range:
            topic_docs=WSM_collapsing(topic_docs,k,word_similarity_matrix)        
            with open(os.path.join(output_folder,dataset,"topics_grouped.txt"),"a+") as f:
                json.dump({"k":k,"topic_rep":topic_docs},f,indent=4)

    def get_topic_rep(self,dataset,output_folder,top_words = 10):
        with open(os.path.join(output_folder,dataset,"topics_grouped.txt"),"r") as f:
            filter_temp_docs=json.load(f)
        tfidf_vectorizer = TfidfVectorizer()
        tfidf = tfidf_vectorizer.fit_transform([" ".join(filter_temp_docs[key]) for key in filter_temp_docs.keys()]) 
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        topic_rep = []
        topic_rep_dic = {}

        for i in range(len(filter_temp_docs.keys())):
            key = list(filter_temp_docs.keys())[i]
            first_document_vector=tfidf[i] 
            feature_array = np.array(tfidf_feature_names)
            tfidf_sorting = np.argsort(first_document_vector.toarray()).flatten()[::-1]
            top_n = feature_array[tfidf_sorting]
            answer = top_n[:top_words]            
            topic_rep.append(answer)
            topic_rep_dic[key] = list(answer)
            
        # return topic_rep, topic_rep_dic
        with open(os.path.join(output_folder,dataset,"topic_rep.json"),"w", encoding='utf-8') as f:
            json.dump(topic_rep_dic,f,indent=4)        
    
    def collapse(self,dataset,k,input_folder,output_folder):
        self.group_topics(dataset,input_folder,output_folder,k)

    def get_rep(self,k,dataset,input_folder,output_folder):    
        self.get_topic_rep(dataset,output_folder)