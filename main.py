from flask import Flask, request, jsonify
import os
from langchain.chat_models import ChatOpenAI
from google.cloud import secretmanager
from pinecone import Pinecone
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set the environment variable
os.environ["FLASK_DEBUG"] = "production"

def access_secret_version(secret_id, version_id="latest"):
    project_id = "helpmum-ai-chatbot"
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] = access_secret_version("pinecone-api-key") 

# Initialize OpenAI API Key
os.environ["OPENAI_API_KEY"] = access_secret_version("openai-api-key") 

# Initialize Chat Model
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-4-turbo-preview',
    temperature = 0.2
)
embed_model = VertexAIEmbeddings(model_name = 'textembedding-gecko@001')

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@app.route('/test', methods=['GET'])
def test():
    return "Chatbot server is running."

@app.route('/chat', methods=['POST'])
def chat_with_bot():
    data = request.json
    user_message = data.get('message')

    if not isinstance(user_message, str):
        return jsonify({'error': 'Message must be a string'}), 400
  
    # Query the Pinecone index for relevant documents
    query_embed = embed_model.embed_query(user_message)
    index_name = 'helpmum-ai-chatbot'  
    pinecone_index = pc.Index(index_name)
    results = pinecone_index.query(vector=query_embed, top_k=2, include_values=True, filter=None, include_metadata=True)

    # Include results from Pinecone in the conversation
    for result in results['matches']:
        doc_content = result['metadata']['additional_info']

   
    prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You're VaxAI, an AI bot specializing in vaccination and immunization only!.\
                Your name is VaxAI, \
            don't ever ever answer any questions outside immunization or vacccination, \
            make sure all your answers are according to the latest WHO guideline and very correct, \
                              respond very briefly and politely, \
                              don't ever ever answer any questions outside immunization or vacccination, \
Please respond concisely, in one short sentence, and directly address the question like in a normal conversation. \
Before you answer, check this document: ["+doc_content+ "] \
    for possible answers and modify to a reasonable response suitable for the question for any gender or person!\
If unsure about an answer, state clearly that you don't know based on the provided context. \
Keep asking questions until you get the answer you need.\
don't ever ever answer any questions outside immunization or vacccination\
Do not use personal information unless explicitly given in the query."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

    conversation = LLMChain(
        llm=chat,
        prompt=prompt,
        verbose=True,
        memory=memory
)

    # Generate the response from the chat model

    res = conversation({"question": user_message})

    response = res['text']

    # Check if "message_response:" is in the response
    if "message_response:" in response:
        # Split the string and take the part after "message_response:"
        actual_response = response.split("message_response: ")[1]
    else:
        # If "message_response:" is not found, use the original response
        actual_response = response

    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(actual_response)

    return jsonify({'response': actual_response})