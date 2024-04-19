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


from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from flask import Flask
import string
import random
from flask_mail import Mail, Message
from sqlalchemy.exc import SQLAlchemyError
from flask_migrate import Migrate



# Initialize the Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://localhost/macbookair'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

class Organization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    api_key = db.Column(db.String(12), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    country = db.Column(db.String(50), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    date = db.Column(db.DateTime(timezone=True), server_default=func.now())

class Messages(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    chat_id = db.Column(db.String(36), nullable=False)  # UUID for chat session
    date = db.Column(db.DateTime(timezone=True), server_default=func.now())

    organization = db.relationship('Organization', backref=db.backref('messages', lazy=True))

def generate_api_key(length=12):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


app.config['MAIL_SERVER'] = 'smtp.example.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@example.com'
app.config['MAIL_PASSWORD'] = 'your-email-password'
mail = Mail(app)

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


@app.route('/signup', methods=['POST'])
def signup_organization():
    data = request.json
    try:
        api_key = generate_api_key()
        # Ensure the API key is unique
        while Organization.query.filter_by(api_key=api_key).first() is not None:
            api_key = generate_api_key()
        
        new_org = Organization(
            api_key=api_key,
            name=data['name'],
            country=data['country'],
            state=data['state'],
            email=data['email']
        )

        print('----------------------Generated API KEY:  ',api_key)
        
        db.session.add(new_org)
        db.session.commit()

        # Send email with API key
        try:
            msg = Message("Your API Key", sender="ajiscomorac@gmail.com", recipients=[data['email']])
            msg.body = f"Your API Key is: {api_key}"
            mail.send(msg)
        except Exception as e:  # Catching a more general exception
            print(f"Failed to send email due to an unexpected error: {e}")
            db.session.rollback()  # Optionally, roll back the database session or handle as needed
            return jsonify({'error': 'Failed to send API key via email due to an unexpected error.'}), 500

    except SQLAlchemyError as e:
        # Handle database errors
        print(f"Database error: {e}")
        db.session.rollback()  # Roll back the session in case of error
        return jsonify({'error': 'Database operation failed.'}), 500

    return jsonify({'message': 'Organization registered successfully, API key sent via email.'}), 201



@app.route('/chat', methods=['POST'])
def chat_with_bot():
    # Extract the API key from the request headers
    api_key = request.headers.get('X-API-KEY')
    print('----------------------API KEY:  ',api_key)
    if not api_key:
        return jsonify({'error': 'API key is missing'}), 401
    
    api_key = str(api_key)

    # Verify the API key
    organization = Organization.query.filter_by(api_key=api_key).first()
    if not organization:
        return jsonify({'error': 'Invalid API key'}), 403

    # Ensure the message is in the request and is a string
    data = request.json
    user_message = data.get('message')
    if not user_message or not isinstance(user_message, str):
        return jsonify({'error': 'Message must be a string and cannot be empty'}), 400
  
    # Query the Pinecone index for relevant documents
    query_embed = embed_model.embed_query(user_message)
    index_name = 'helpmum-ai-chatbot'  
    pinecone_index = pc.Index(index_name)
    results = pinecone_index.query(vector=query_embed, top_k=2, include_values=True, filter=None, include_metadata=True)

    # Process results from Pinecone for conversation
    doc_contents = [result['metadata']['additional_info'] for result in results['matches']]
    doc_content = doc_contents[0] if doc_contents else "No additional info found."

    # Constructing the prompt for the chat model
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                f"You're an AI bot specializing in vaccination and immunization only! "
                f"Please respond concisely, in one short sentence, and directly address the question like in a normal conversation. "
                f"Before you answer, check this document: [{doc_content}] for possible answers and modify to a reasonable response suitable for the question for any gender or person! "
                f"If unsure about an answer, state clearly that you don't know based on the provided context. "
                f"Keep asking questions until you get the answer you need. "
                f"Do not use personal information unless explicitly given in the query."
            ),
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

    # Extract actual response
    if "message_response:" in response:
        actual_response = response.split("message_response: ")[1]
    else:
        actual_response = response

    # Update chat memory
    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(actual_response)

    return jsonify({'response': actual_response})

if __name__ == "__main__":
    app.run(debug=False)

