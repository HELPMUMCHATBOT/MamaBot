import os
from google.cloud import secretmanager
from pinecone import Pinecone
from openai.error import RateLimitError
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import re
from openai.error import RateLimitError
from langchain.document_loaders import CSVLoader


def access_secret_version(secret_id, version_id="latest"):
    project_id = "helpmum-ai-chatbot"
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Function to split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] = access_secret_version("pinecone-api-key") 

embed_model = VertexAIEmbeddings(model_name = 'textembedding-gecko@001')

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Check if the index already exists and create if not
index_name = 'helpmum-ai-chatbot'  

pinecone_index = pc.Index(index_name)


def parse_retry_wait_time(error_message):
    """
    Parse the wait time from the rate limit error message.
    """
    match = re.search(r'Please try again in (\d+)s', error_message)
    if match:
        return int(match.group(1))
    return 60  # Default wait time


def batch_embed_documents_with_error_handling(embed_model, documents, batch_size=10):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_contents = [doc.page_content for doc in batch]  # Extracting string content
        progress = (i + batch_size) / len(documents) * 100
        print(f"-------------------------------------\nEmbed Progress: {int(progress)}%  --  Size: {i+batch_size}\n-------------------------------------")
        try:
            for attempt in range(3):  # Retry logic with up to 3 attempts
                try:
                    batch_embeddings = embed_model.embed_documents(batch_contents)
                    for embedding, content in zip(batch_embeddings, batch_contents):
                        yield embedding, content  # Yielding each embedding and content
                    break  # Break the loop if successful
                except RateLimitError as e:
                    wait_time = parse_retry_wait_time(str(e))
                    print(f"Rate limit reached on attempt {attempt+1}. Waiting for {wait_time} seconds. Details: {e}")
                    time.sleep(wait_time)
                except Exception as e:
                    print(f"Error while processing batch on attempt {attempt+1}: {e}")
                    if attempt == 2:  # On last attempt, re-raise to indicate failure
                        raise
        except Exception as e:
            print(f"Final error after retries: {e}")


csv_file_path = 'final_df_merged_col.csv'
text_column_name = 'message_response'

# Load documents from CSV file
csv_loader = CSVLoader(csv_file_path, text_column_name)
documents = csv_loader.load()

# Split documents into chunks
docs = split_docs(documents)

# Generate embeddings for chunks
all_embeddings = []
all_batch_content = []
for embedding, content in batch_embed_documents_with_error_handling(embed_model, docs):
    all_embeddings.append(embedding)
    all_batch_content.append(content)

print(f"\n\n**************************************\nEmbeding done !!!\n**************************************\n\nPincone vectore storage started ....\n")
        
for idx, (embed, doc) in enumerate(zip(all_embeddings, all_batch_content)):
    vector_id = str(idx)
    vector_values = embed 
    metadata = {"additional_info": doc}
    pinecone_index.upsert(vectors=[(vector_id, vector_values, metadata)])

print(f"\n**************************************\nPincone vectore storage complete !!!\n**************************************\n")