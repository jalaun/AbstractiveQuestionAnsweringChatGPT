import os
import pinecone
import pandas as pd

YOUR_API_KEY = "expired :("
YOUR_ENVIRONMENT = "northamerica-northeast1-gcp"
index_name = "abstractive-question-answering"
total_doc_count = 50
batch_size = 64

# Check if the processed data file exists
if os.path.exists('data.csv'):
    # Load the data from the file
    df = pd.read_csv('data.csv')
else:
    # Load and process the data
    wiki_data = load_dataset('vblagoje/wikipedia_snippets_streamed', split='train', streaming=True).shuffle(seed=960)
    history = wiki_data.filter(lambda d: d['section_title'].startswith('History'))

    docs = []
    for i, d in enumerate(tqdm(history, total=total_doc_count)):
        doc = {
            "article_title": d["article_title"],
            "section_title": d["section_title"],
            "passage_text": d["passage_text"]
        }
        docs.append(doc)
        if i >= total_doc_count:
            break

    df = pd.DataFrame(docs)
    df.to_csv('data.csv', index=False)

pinecone.init(api_key=YOUR_API_KEY, environment=YOUR_ENVIRONMENT)

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric="cosine")

index = pinecone.Index(index_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)

for i in tqdm(range(0, len(df), batch_size)):
    i_end = min(i+batch_size, len(df))
    batch = df.iloc[i:i_end]
    emb = retriever.encode(batch["passage_text"].tolist()).tolist()
    meta = batch.to_dict(orient="records")
    ids = [f"{idx}" for idx in range(i, i_end)]
    to_upsert = list(zip(ids, emb, meta))
    _ = index.upsert(vectors=to_upsert)

tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)

def query_pinecone(query, top_k):
    xq = retriever.encode([query]).tolist()
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

def format_query(query, context):
    context = [f"<P> {m['metadata']['passage_text']}" for m in context]
    context = " ".join(context)
    query = f"question: {query} context: {context}"
    return query

def generate_answer(query):
    inputs = tokenizer([query], max_length=1024, return_tensors="pt")
    ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=20, max_length=40)

def generate_answer(query):
    inputs = tokenizer([query], max_length=1024, return_tensors="pt")
    ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=20, max_length=40)
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return print(answer)

# Testing helper functions
query = "Who was Julius Caesar?"
result = query_pinecone(query, top_k=1)
query = format_query(query, result["matches"])
generate_answer(query)