import argparse
from models import *
from utils import *

output_folder="./outputs"
input_folder="./input"

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-model', help='chatGPT\llama',choices=['chatGPT','llama'],required=True,default='chatGPT',type=str)
    parser.add_argument('-dataset', help='dataset to run topic modeling(TdefMemes\FB_hateful_memes\Memotion)',required=True,default="TDefMemes",type=str)
    parser.add_argument('-merging', help='merging technique(pbm\wsm)',required=False,default="False",type=str)
    parser.add_argument('-k_range', help='range of topk',required=False,default="10,20,30,40,50",type=str)
    args = parser.parse_args()
    
    dataset=args.dataset
    model=args.model
    k_range=[int(item) for item in args.k_range.split(",")]
    merging=args.merging

    if model=="chatGPT" and merging=="pbm":
        model=ChatGPT()
    elif model=="chatGPT" and merging=="wsm":
        model=ChatGPT_Base()
    elif model=="llama":
        model=Llama()

    # collect topics
    model.collect_topics(dataset,input_folder,output_folder)
    # collapse topics
    model.collapse(dataset,k_range,input_folder,output_folder)
    # get topic representation
    model.get_rep(k_range,dataset,input_folder,output_folder)
    # calculate metrics
    calc_metrics(dataset,input_folder,output_folder,k_range)
    