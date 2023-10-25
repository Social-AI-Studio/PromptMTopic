from .base_model import Base_Model
import os
import json
import yaml
from utils import *
from api_utils import *
from ctfidf import *
import ast
import tqdm

class ChatGPT_Base(Base_Model):
    def __init__(self) -> None:
        super().__init__()
        with open("config.yaml","r") as f:
            self.config=yaml.safe_load(f)
        self.collect_template=self.config['chatGPT']['collect_template']
        self.collect_examples=ast.literal_eval(self.config['chatGPT']['collect_examples'])

    def callChatGPT(self,data,covered,output_path):
        failed={}
        for file,data in tqdm.tqdm(data.items()):
            if(file in covered):
                continue
            prompt_kwargs = {
            "caption": data["caption"],
            "text": data["text"]
            }
            prompt=self.collect_template.format(**prompt_kwargs)
            prompt=[*self.collect_examples,{
            "role":"user","content":prompt
            }]      
            try:  
                response=retry_callAPI("", msgs=prompt)
                response=self.extract_response(response)
            except:
                failed.update({file:data})
                continue
            with open(output_path,"a+") as f:
                json.dump({file:response},f)
                f.write("\n")    
        return failed
    
    def collect_topics(self,dataset,input_folder,output_folder): 
        with open(os.path.join(input_folder,dataset,"captions.json"),"r") as f:
            file_captions=json.load(f)
        with open(os.path.join(input_folder,dataset,"text.json"),"r") as f:
            file_text=json.load(f)   
        data={file:{"caption":file_captions[file],\
                    "text":text} for file,text in file_text.items() if file in file_captions}
        
        output_dir=os.path.join(output_folder,dataset)
        os.makedirs(output_dir, exist_ok=True)
        output_path=os.path.join(output_dir,"document_topics.txt")

        covered=[]
        if(os.path.exists(output_path)):
            with open(output_path,"r") as f:
                covered=f.read().split("\n")
            covered=[item for item in covered if item!='']    
            covered={k:v for item in covered for k,v in ast.literal_eval(item).items()}
            covered=[file for file,_ in covered.items()]    
                 
        print("calling chatGPT to extract topics from dataset")
        failed=self.callChatGPT(data,covered,output_path)       
        while(len(failed)!=0):
            failed=self.callChatGPT(failed,covered,output_path)

    def read_topics(self,dataset,output_folder):
        topic_files={}
        topic_files["inappropriate"]=[]
        topic_files["miscellaneous"]=[]
        topics_file=os.path.join(output_folder,dataset,"document_topics.txt")
        with open(topics_file,"r") as f:
            doc_topics=f.read().split("\n")
        for item in doc_topics:
            if(item.strip()==''):
                continue
            for file, topics in ast.literal_eval(item).items():
                try:
                    temp=ast.literal_eval(topics)
                    for topic in temp:
                        topic=topic.lower().strip()
                        if topic in topic_files:
                            topic_files[topic].append(file)
                        else:
                            topic_files[topic]=[file]    
                except:
                    if(any([adj in topics.lower() for adj in ["offensive","inappropriate","derogatory","not appropriate"]])):
                        topic_files["inappropriate"].append(file)
                    else:    
                        topic_files["miscellaneous"].append(file)
        if(len(topic_files["miscellaneous"])==0):
            del topic_files["miscellaneous"]
        if(len(topic_files["inappropriate"])==0):
            del topic_files["inappropriate"]                        
        topic_files={topic:list(set(files)) for topic,files in topic_files.items()}                
        all_topics={topic:len(files) for topic,files in topic_files.items()}  
        all_topics=sorted(all_topics.items(), key = lambda x:x[1], reverse = True)
        all_topics=[item[0] for item in all_topics]
        topic_files={topic.lower():topic_files[topic] for topic in all_topics}
        return topic_files
    
    def collapse(self, dataset, k, input_folder, output_folder):
        topic_docs=self.read_topics(dataset,output_folder)
        return super().collapse(topic_docs, dataset, k, input_folder, output_folder)
    
class ChatGPT(ChatGPT_Base):
    def __init__(self) -> None:
        super().__init__()
        
        self.group_template=self.config['chatGPT']['group_template']
        self.group_examples=ast.literal_eval(self.config['chatGPT']['group_examples'])
        
    def extract_response(self,response):
        response=response["choices"][0]["message"]["content"]
        return response

    def group(self,topics,top_topics,output_path):
        index=0
        while(index<len(topics)):
            print("calling ChatGPT to map topics.",end="\r")        
            if topics is not None:
                prompt_kwargs = {
                "topics" : "\n".join(top_topics),    
                "topic": topics[index]
                }
            else:
                prompt_kwargs = {
                "topics" : "\n".join(top_topics[index+1:]),    
                "topic": top_topics[index]
                }
            
            prompt=self.group_template.format(**prompt_kwargs)
            prompt=[*self.group_examples,{
                "role":"user","content":prompt
                }] 
            
            response=retry_callAPI('',prompt)
            content=self.extract_response(response)
                
            with open(output_path,"a+") as f:            
                json.dump({topics[index]:content},f) 
                f.write("\n")               
            index+=1

    def group_topics(self,topic_docs,dataset,output_folder):
        output_dir=os.path.join(output_folder,dataset)
        os.makedirs(output_dir, exist_ok=True)
        output_path= os.path.join(output_folder,dataset,"topics_grouped.jsonl")
        
        all_topics={topic:len(files) for topic,files in topic_docs.items()}  
        all_topics=sorted(all_topics.items(), key = lambda x:x[1], reverse = False)
        all_topics=[item[0] for item in all_topics]
        topic_docs={topic:list(set(topic_docs[topic])) for topic in all_topics} 

        all_topics=[topic for topic,_ in topic_docs.items()]
        top_topics=all_topics[-200:]
        all_topics=all_topics[:-200]
        self.group(all_topics,top_topics,output_path)
        self.group(None,top_topics,output_path)

    def map_topics_to_k(self,topic_docs,dataset,top_k_range,output_folder):
        mapping={}
        all_topics={topic:len(files) for topic,files in topic_docs.items()}  
        all_topics=sorted(all_topics.items(), key = lambda x:x[1], reverse = False)
        all_topics=[item[0] for item in all_topics]
        group_file=os.path.join(output_folder,dataset,"topics_grouped.jsonl").format(output_folder,dataset)
        with open(group_file,"r") as f:
            groups=f.read().split("\n")
        groups={k.lower():v.lower() for item in groups for k,v in ast.literal_eval(item).items()}
        groups={k:v[:-1] if v[-1]=='.' else v for k,v in groups.items()}
        groups={k:[topic for topic in list(topic_docs.keys()) if topic in v] for k,v in groups.items()}
        groups={k:v if k!="miscellaneous" else ["miscellaneous"] for k,v in groups.items()}
        groups={k:topics[0] if len(topics)>0 else None for k,topics in groups.items()}

        mapping_k={}
        for top_k in top_k_range:
            groups={k:v for k,v in groups.items() if k in all_topics[:(len(topic_docs)-top_k)]}
            for k,topic in groups.items():
                if topic is None:
                    mapping[k]="miscellaneous"
                    continue
                temp=topic
                iter=0
                while temp in groups and iter<20:
                    temp=groups[temp]
                    iter+=1
                if(iter==20 or temp is None):
                    # print(k)
                    mapping[k]="miscellaneous"  
                else:       
                    mapping[k]=temp                
            mapping_ordered={topic:mapping[topic] for topic in all_topics if topic in mapping}
            with open(os.path.join(output_folder,dataset,"mapping_{}.json".format(top_k)),"w") as f:
                json.dump(mapping_ordered,f) 
            mapping_k[top_k]=mapping_ordered        
        return mapping_k    
    
    def map_files_to_groups(self,topic_docs,mappings,dataset,output_folder):
        for top_k,mapping in mappings.items():
            for k,topic in mapping.items():
                # if topic_docs[topic] is not None:
                if k==topic:
                    continue
                topic_docs[topic].extend(topic_docs[k])
                topic_docs.pop(k)
                # else:
                #     topic_docs[topic]=topic_docs[k]  
            topic_docs={topic:list(set(docs)) for topic,docs in topic_docs.items()}
            with open(os.path.join(output_folder,dataset,"grouped_ktopic_docs_{}.json".format(top_k)),"w") as f:
                json.dump(topic_docs,f)

    def get_topic_rep(self,k_range,dataset,input_folder,output_folder):
        with open(os.path.join(output_folder,dataset,"topic_rep.txt"),"w") as f:
            for k in k_range: 
                with open(os.path.join(output_folder,dataset,"grouped_ktopic_docs_{}.json".format(k)),"r") as f:
                    topic_docs=json.load(f)
                json.dump({"k":k,"topic_rep":self.collect_topic_rep(topic_docs,dataset,input_folder)},f)
                f.write("\n")        

    def order_topic_rep(self,dataset,output_folder):
        with open(os.path.join(output_folder,dataset,"topic_rep.txt"),"r") as f:
            topic_rep=f.read().split("\n")
        topic_rep=[item for item in topic_rep if item!='']
        # topic_rep=topic_rep[-1]
        # topic_rep=ast.literal_eval(topic_rep)['topic_rep']
        topic_rep=[json.loads(item) for item in topic_rep]

        covered=[]
        output_path=os.path.join(output_folder,dataset,"topic_rep_ordered.txt")
        if os.path.exists(output_path):
            with open(output_path,"r") as f:
                covered=f.read().split("\n")
            covered=[item for item in covered if item!=''] 
            if(len(covered)>0):
                covered=[json.loads(item) for item in covered]
                covered=[item['topic'] for item in covered]

        for item in topic_rep:
            index=0    
            print("calling ChatGPT to order words for topics")
            for topic,words in tqdm.tqdm(item["topic_Rep"].items()):
                if(topic in covered):
                    index+=1
                    continue
                prompt_kwargs={"words":"\n".join([" {}. {}".format(i,word) for i,word in enumerate(words[:100])]),"topic":topic}
                prompt=self.word_template.format(**prompt_kwargs)
                prompt=[*self.word_order_examples,{
                    "role":"user","content":prompt
                    }] 
                response=retry_callAPI('',prompt)
                response=self.extract_response(response)
                
                with open(os.path.join(output_folder,dataset,"topic_rep_ordered_{}.txt".format(item["k"])),"a+") as f:
                    json.dump({"topic":topic,"words":response},f)
                    f.write("\n")
                index+=1     

    def clean_topic_rep(self,k_range,dataset,output_folder):
        with open(os.path.join(output_folder,dataset,"topic_rep.txt"),"r") as f:
            original_rep=f.read().split("\n")
        original_rep=[item for item in original_rep if item!='']    
        # original_rep=original_rep[-1]
        original_rep=ast.literal_eval(original_rep)['topic_rep']
        
        for k in k_range:
            with open(os.path.join(output_folder,dataset,"topic_rep_ordered_{}.txt".format(k)),"r") as f:
                topic_rep=f.read().split("\n")
            topic_rep=[item for item in topic_rep if item!='']    
            
            rep_cleaned=[ast.literal_eval(item) for item in topic_rep]
            rep_cleaned={item['topic']:item['words'] for item in rep_cleaned}

            rep_cleaned={topic:[word for word in words if word.lower().strip() in original_rep[topic]] for topic,words in rep_cleaned.items()}
            rep_cleaned={topic:list(dict.fromkeys(words)) for topic,words in rep_cleaned.items()}
            rep_cleaned={topic:words if len(words)>0 else original_rep[topic][:10] for topic,words in rep_cleaned.items()}
            rep_cleaned={topic:words if len(words)==10 else words+[token for token in original_rep[topic] if token not in words][:10-len(words)] \
                        for topic,words in rep_cleaned.items()}

            with open(os.path.join(output_folder,dataset,"topic_rep_ordered_{}.txt".format(k)),"w") as f:
                json.dump(rep_cleaned,f)    

    def collapse(self,dataset,k,input_folder,output_folder):
        topic_docs=self.read_topics(dataset,output_folder)
        self.group_topics(topic_docs,dataset,output_folder)
        mapping=self.map_topics_to_k(topic_docs,dataset,k,output_folder)
        self.map_files_to_groups(topic_docs,mapping,dataset,output_folder)

    def get_rep(self,k,dataset,input_folder,output_folder):    
        self.get_topic_rep(k,dataset,input_folder,output_folder)
        self.order_topic_rep(dataset,output_folder)
        self.clean_topic_rep(dataset,k,output_folder)
