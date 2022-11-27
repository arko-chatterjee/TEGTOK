from datasets import load_dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from annoy import AnnoyIndex
import torch
import time

EMBEDDING_DIM = 768
TOP_K = 5

current = time.time()
def measure():
    global current
    t = time.time()
    print(t - current)
    current = t

print("LOADING DATA")
dataset = load_dataset("wiki_dpr", "psgs_w100.nq.compressed", cache_dir="~/Downloads/huggingface/datasets")

# print("LOADING MODEL")
# tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
# model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

print("INDEXING DATA")
indexer = AnnoyIndex(EMBEDDING_DIM, 'dot')
for i in range(0, dataset['train'].num_rows, 5):
    if (i % 100000 == 0):
        print(i)
        measure()
    v = dataset['train'][i]['embeddings']
    indexer.add_item(int(i/5), v)

print("BUILDING TREE")
indexer.build(10) # 10 trees
measure()
print("WRITING TREE")
indexer.save('wikipedia.ann')
measure()