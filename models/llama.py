import transformers
from base_model import Base_Model
import torch
import os
import json
import ast

class Llama(Base_Model):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained("yahma/llama-13b-hf")
        self.model = transformers.LlamaForCausalLM.from_pretrained("yahma/llama-13b-hf", device_map="sequential", torch_dtype=torch.float16)

    def call_llama(self,prompt, max_token=50, device = "cuda:0"):
        # Tokenize the prompt
        batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        # Move the batch to the first GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        generated = self.model.generate(batch["input_ids"], max_new_tokens=max_token)
        # Decode the generated output
        decoded_output = self.tokenizer.decode(generated[0])
        return decoded_output
    
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
        failed=self.call_llama(data,covered,output_path)       
        while(len(failed)!=0):
            failed=self.call_llama(failed,covered,output_path)

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
                    temp=topics.strip().split("  ")[0]
                    temp=temp.split(',')
                    temp=[item.lstrip('\'').rstrip('\'') for item in temp] 
                    
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