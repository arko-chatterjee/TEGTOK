from datasets import load_dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from annoy import AnnoyIndex
import torch
import time
import json

EMBEDDING_DIM = 768

current = time.time()
def measure():
    global current
    t = time.time()
    print(t - current)
    current = t

print("LOADING MODEL")
tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

def createAnnoyFor(fileName):
    print("LOADING DATA")
    file = open('../data/'+fileName+'.json')     # SELECT FILE
    redditData = json.load(file)
    comments = []

    print("INDEXING DATA")
    indexer = AnnoyIndex(EMBEDDING_DIM, 'dot')
    i = 0
    for post in redditData:
        for comment in post['comments']:
            if(len(comment) < 512):
                if (i % 1000 == 0):
                    print(i)
                    measure()
                comments.append(comment)
                embedding = encode_question(comment)
                indexer.add_item(i, embedding)
                i += 1
    print("BUILDING TREE")
    indexer.build(10) # 10 trees
    measure()
    print("WRITING TREE")
    indexer.save(fileName+'.ann')
    measure()
    print("WRITING JSON")
    json_object = json.dumps(comments, indent=4)
    with open(fileName+"Comments.json", "w") as outfile:
        outfile.write(json_object)
    measure()


def encode_question(q):
    with torch.no_grad():
        input_ids = tokenizer(q, return_tensors="pt")["input_ids"]
        embeddings = model(input_ids).pooler_output.numpy()[0]
        return embeddings
    
if __name__ == '__main__':
    createAnnoyFor("RoastMe")
    createAnnoyFor("FreeCompliments")