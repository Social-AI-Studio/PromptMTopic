
# PromptMTopic

Official implementation for paper [Prompting Large Language Models for Unsupervised Multimodal Meme Topic Modeling]()


## Setup
1. Clone the repo 
``` bash
git clone https://github.com/Social-AI-Studio/PromptMTopic.git 
pip install -r requirements.txt
```
2. If run with OpenAI models, set up OpenAI API key and intialize `openai.api_key` in `api_utils.py`
If run with LLaMA, download weights from [huggingface](https://huggingface.co/huggyllama/llama-13b)

## Usage
### Prepare Input

Prepare ```input``` folder. This contains subfolders of datasets you run the model on. Each dataset subfolder contains two json files below, each with key as image name and value as corresponding caption/text.
-  ```captions.json``` 
- ```text.json```
- 
Concatenate captions and OCR text in a single line for each sample and save in a 'corpus' directory inside input folder for each dataset used. This is required by the evaluation library.

In the paper, we removed text from the images then use BLIP-2 for captioning.

Change the path of ```input``` and ```output``` folders to your path in ```config.yaml```.

### Generating Topics and Evaluation
Insert corresponding arguments in ```< >```

```bash
python3 promptMtopic.py -model <model> -dataset <dataset> -merging <wsm or pbm> -k_range <range of generated topics>
```
For example:
```bash
python3 promptMtopic.py -model chatGPT -dataset TDefMemes -merging pbm -k_range 10,20,30,40,50
```

## Datasets
The topic models have been evaluated on the following datasets:
| Dataset                        | #Memes
| -------------------------------| -------------  |
| Facebook Hateful Memes(FHM)    | 10,000         |
| Total Defence Memes(TDefMeme)  | 2,513          |
| Memotion                       | 6,992          |

## Acknowledgements

 - Topic Metric: [Preferred.AI's Topic Metric](https://github.com/PreferredAI/topic-metrics/tree/main)
 - Captioning Module: [LAVIS's BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
 - ctfidf: [BERTopic](https://github.com/MaartenGr/BERTopic/blob/62e97ddea6cdcf9e4da25f9eaed478b22a9f9e20/bertopic/vectorizers/_ctfidf.py#L4) 
