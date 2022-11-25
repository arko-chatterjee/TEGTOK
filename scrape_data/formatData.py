from datasets import load_dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from annoy import AnnoyIndex
import torch
import time
import json

TOP_K = 5
EMBEDDING_DIM = 768

current = time.time()
def measure(s = ""):
    global current
    t = time.time()
    print(s, t - current)
    current = t

print("LOADING MODEL")
tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
measure("model load time")

print("LOADING WIKIPEDIA DATA")
indexer = AnnoyIndex(EMBEDDING_DIM, 'dot')
indexer.load('wikipedia.ann') # Will just mmap the file, super fast

dataset = load_dataset("wiki_dpr", "psgs_w100.nq.compressed", cache_dir="~/Downloads/huggingface/datasets")
measure("wikipedia data load time")

def createDatasetFor(fileName):
    print("LOADING REDDIT DATA")
    file = open('../data/'+fileName+'.json')     # SELECT FILE
    redditData = json.load(file)

    redditIndex = AnnoyIndex(EMBEDDING_DIM, 'dot')
    redditIndex.load(fileName+'.ann')

    commentFile = open(fileName+'Comments.json')
    redditComments = json.load(commentFile)
    measure("reddit data load time")

    print("CREATING DATASET")
    i = 0
    output = []
    for post in redditData:
        if (i % 100 == 0):
            measure()
        i += 1
        question_embedding = encode_question(post['title'])
        indsW = indexer.get_nns_by_vector(question_embedding, TOP_K, include_distances=True)
        indsR = redditIndex.get_nns_by_vector(question_embedding, TOP_K, include_distances=True)
        textArrW = []
        titleArrW = []
        scoreArrW= []
        textArrR = []
        titleArrR = []
        scoreArrR = []
        for (index, distance) in zip(indsW[0], indsW[1]):
            textArrW.append(dataset['train'][index]['text'])
            titleArrW.append(dataset['train'][index]['title'])
            scoreArrW.append(distance)
        for (index, distance) in zip(indsR[0], indsR[1]):
            textArrR.append(redditComments[index])
            titleArrR.append("")
            scoreArrR.append(distance)
        for comment in post['comments']: 
            output.append({"post": post['title'], "response": comment, 
                "wiki_knowledge": textArrW, "wiki_knowledge_title": titleArrW, "wiki_knowledge_score": scoreArrW,
                "reddit_knowledge": textArrR, "reddit_knowledge_title": titleArrR, "reddit_knowledge_score": scoreArrR,
            })

    print("WRITING JSON")
    json_object = json.dumps(output, indent=4)
    with open(fileName+"Augmented.json", "w") as outfile:
        outfile.write(json_object)
    measure()



def encode_question(q):
    with torch.no_grad():
        input_ids = tokenizer(q, return_tensors="pt")["input_ids"]
        embeddings = model(input_ids).pooler_output.numpy()[0]
        return embeddings
    
if __name__ == '__main__':
    print('ENCODING DATA')
    createDatasetFor('RoastMe')
    createDatasetFor('FreeCompliments')
    