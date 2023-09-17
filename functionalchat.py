import psycopg2
from sqlalchemy import create_engine
from langchain import LLMMathChain, OpenAI, SelfAskWithSearchChain, SerpAPIWrapper, SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from SQLDatabaseChain import SQLDatabaseChain
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import Pinecone
from langchain.memory.chat_memory import BaseMemory
from langchain.prompts import ChatMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from SystemMessageContent import SystemMessageContent
from langchain.prompts.base import BasePromptTemplate as PromptTemplate
import uuid
from datetime import datetime, timedelta
from psycopg2.extras import Json
import subprocess
import socket
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from langchain.memory.utils import get_prompt_input_key
import logging
import openai
import json
from collections import deque
from langchain import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Setup logger
logger = logging.getLogger('MyApplication')
logger.setLevel(logging.DEBUG)

# Create a formatter with timestamp
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console Handler with Debug level
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
c_handler.setFormatter(formatter)

# File Handler with Debug level and rotation
f_handler = logging.handlers.RotatingFileHandler('applog.log', maxBytes=1048576, backupCount=5)
f_handler.setLevel(logging.DEBUG)
f_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.info("Welcome to the Core Nexus.")

# Read environment variables from the file
with open("C:\\Users\\Tyvon\\Dev\\Environment_Variables.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.strip():
            key, value = line.strip().split('=')
            os.environ[key] = value

# API keys setup
API_KEYS = {
    "OPENAI": os.environ.get("OPENAI_API_KEY"),
    "PINECONE": os.environ.get("PINECONE_API_KEY"),
    "SERPAPI": os.environ.get("SERPAPI_API_KEY")
}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
openai.api_key = OPENAI_API_KEY

for key, value in API_KEYS.items():
    if value is not None:
        os.environ[f"{key}_API_KEY"] = value

embeddings = OpenAIEmbeddings()

# Database connection
db_conn = None
engine = None

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def start_postgres():
    # Check if PostgreSQL is already running
    if not is_port_open(5432):
        print("PostgreSQL is already running.")
        return

    try:
        # Specify the path to the bin directory of PostgreSQL
        bin_path = "C:\\Program Files\\PostgreSQL\\15\\bin"
        
        # Specify the path to your PostgreSQL data directory
        data_path = "C:\\Program Files\\PostgreSQL\\15\\data"
        
        # Command to start PostgreSQL
        command = f'{bin_path}\\pg_ctl start -D "{data_path}"'
        
        # Run the command
        subprocess.run(command, shell=True, check=True)
        
        print("PostgreSQL server started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while starting PostgreSQL: {e}")

# Call the function to start PostgreSQL
start_postgres()

def get_db_connection():
    global db_conn
    if db_conn is None or db_conn.closed:
        try:
            db_conn = psycopg2.connect(
                database="postgres",
                user="postgres",
                password=os.environ.get("POSTGRES_PASSWORD"),
                host="localhost",
                port="5432"
            )
        except Exception:
            return None
    return db_conn

def get_db_engine():
    global engine
    if engine is None:
        try:
            engine = create_engine(
                "postgresql+psycopg2://postgres:Vbevbe1!@localhost:5432/postgres"
            )
        except Exception:
            return None
    return engine

def record_user_interaction(user_id, interaction_info):
    try:
        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()


        # Extract details from interaction_info
        user_input = interaction_info.get('message')
        agent_response = interaction_info.get('response')
        context = interaction_info.get('context')
        embedding_vector_metadata = interaction_info.get('embedding_vector_metadata') 

        # Adjusted SQL query to match the database schema
        query = """INSERT INTO conversation_history 
                   (user_id, session_id, created_at, message, response, context, embedding_vector) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s)"""

        db_conn = get_db_connection()
        cursor = db_conn.cursor()

        # Adapt Python dict to JSON for PostgreSQL
        cursor.execute(query, (user_id, session_id, created_at, user_input, agent_response, context, Json(embedding_vector_metadata)))  

        db_conn.commit()
    except Exception as e:
        if 'no partition of relation "conversation_history" found for row' in str(e):
            db_conn.rollback()
            create_new_partition(created_at)
            record_user_interaction(user_id, interaction_info)
        else:
            db_conn.rollback()
            logger.error(f"Failed to record user interaction: {e}")
    finally:
        cursor.close()

def create_new_partition(timestamp):
    cursor = None
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor()

        year = timestamp.year
        partition_name = f"conversation_history_y{year}"
        start_range = f"'{year}-01-01'"
        end_range = f"'{year + 1}-01-01'"

        cursor.execute(f"SELECT to_regclass('public.{partition_name}');")
        if cursor.fetchone()[0] is None:
            # SQL command to create a new partition for the conversation_history table
            cursor.execute(
                f"""
                CREATE TABLE public.{partition_name} PARTITION OF public.conversation_history 
                FOR VALUES FROM ({start_range}) TO ({end_range});
                """
            )
            db_conn.commit()
        else:
            db_conn.commit()
    except Exception as e:
        if db_conn:
            db_conn.rollback()
        logger.error(f"Failed to create new partition: {e}")
    finally:
        if cursor:
            cursor.close()

def store_conversation_history(user_id, interaction_info):
    """Stores the conversation history in the PostgreSQL database.

    Parameters:
    user_id (str): The ID of the user.
    interaction_info (dict): Information about the interaction to be stored.

    Returns:
    bool: True if the operation was successful, False otherwise.
    """
    try:
        # Getting the current time to store with the message
        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        # Extract details from interaction_info
        user_input = interaction_info.get('user_input')
        agent_response = interaction_info.get('agent_response')
        context = interaction_info.get('context')
        embedding_vector_metadata = interaction_info.get('embedding_vector_metadata') 

        # SQL query to insert the data into the conversation_history table
        query = """INSERT INTO conversation_history 
                   (user_id, session_id, created_at, user_input, agent_response, context, embedding_vector_metadata) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s)"""

        # Getting a new database connection
        db_conn = get_db_connection()
        cursor = db_conn.cursor()

        # Executing the SQL query with the data to be inserted
        cursor.execute(query, (user_id, session_id, created_at, user_input, agent_response, context, Json(embedding_vector_metadata)))  

        # Committing the transaction to save the changes
        db_conn.commit()

        # Returning True to indicate that the operation was successful
        return True
    except Exception as e:
        # If an exception occurs, rollback the transaction and log the error message
        db_conn.rollback()

        if 'no partition of relation "conversation_history" found for row' in str(e):
            create_new_partition(cursor, db_conn, created_at)
            store_conversation_history(user_id, interaction_info)
        else:
            # Log the error (consider using a logger to log the error message)
            print(f"Failed to record user interaction: {e}")

        # Returning False to indicate that the operation was unsuccessful
        return False
    finally:
        # Closing the cursor to release database resources
        cursor.close()

# A dictionary to store file contents temporarily
file_contents = {}

def read_and_analyze_file(file_path):
    response_dict = {"success": False, "result": None}
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # Now send this content to GPT-4 for analysis
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Analyze this file and Understand the Who, What, Where, When, Why and How."},
                {"role": "user", "content": file_content},
            ]
        )
        
        # Extract the GPT-4 analysis result from the API response
        analysis_result = response['choices'][0]['message']['content']
        
        response_dict["success"] = True
        response_dict["result"] = analysis_result
        
    except Exception as e:
        response_dict["result"] = str(e)
    
    return str(response_dict)  # Return the string representation of the response_dict

def get_conversation_history(user_id, interaction_id=None):
    cursor = None
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor()

        # Adjusted SQL query to fetch the conversation history based on the user_id
        query = """SELECT * FROM conversation_history WHERE user_id = %s"""
        params = (user_id,)
        if interaction_id:
            # Adjusted SQL query to additionally filter based on the interaction_id
            query = """SELECT * FROM conversation_history WHERE user_id = %s AND interaction_id = %s"""
            params = (user_id, interaction_id)
        
        cursor.execute(query, params)
        history = cursor.fetchall()
        db_conn.commit()
        return history
    except Exception as e:
        if db_conn:
            db_conn.rollback()
        logger.error(f"Failed to fetch conversation history: {e}")
        return []
    finally:
        if cursor:
            cursor.close()

def context_enrichment(user_id, interaction_id=None):
    history = get_conversation_history(user_id, interaction_id)
    enriched_context = " ".join([f"{entry[0]} {entry[1]}" for entry in history if entry[0] and entry[1]])
    return enriched_context

class PineconeManager:
    def __init__(self, model="text-embedding-ada-002", index_name="core-nexus-index"):
        self.model = model
        self.openai_embeddings = OpenAIEmbeddings(model=self.model)
        self.index_name = index_name
        self.index = None
        self.initialize_pinecone()
        self.ensure_index_exists()


    def initialize_pinecone(self):
        try:
            pinecone.init(api_key=API_KEYS["PINECONE"], environment='asia-southeast1-gcp-free')
        except Exception:
            pass

    def ensure_index_exists(self):
        try:
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(name=self.index_name, metric="cosine", shards=1)
            self.index = pinecone.Index(index_name=self.index_name)
        except Exception:
            pass
            
    def determine_dynamic_top_k(self, user_input: str) -> int:
        """
        Dynamically determine the value of top_k based on the user_input or other criteria.
        """
        if len(user_input.split()) > 10:
            return 5
        else:
            return 3

    def get_embedding(self, text: Union[str, List[str]]) -> tuple:

        if isinstance(text, list) and isinstance(text[0], HumanMessage):
            text = text[0].content
        elif isinstance(text, HumanMessage):
            text = text.content

        if not text.strip():
            return None, None

        try:
            embedding = self.openai_embeddings.embed_query(text)

            if not isinstance(embedding, list):
                embedding = list(embedding)

            metadata = {'original_text': text}
            return embedding, metadata
        except Exception:
            return None, None

    def save_embedding_to_pinecone(self, text, context_type, additional_metadata=None):
        try:
            embedding_vector, metadata = self.get_embedding(text)
            if embedding_vector:
                vector_id = f"vector-{datetime.utcnow().isoformat()}"
                metadata.update({
                    "context_type": context_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "interaction_type": "question" if "?" in text['input_text'] else "statement",  # Add interaction type based on the text content
                    # ... potentially add more metadata fields here ...
                })
                
                if additional_metadata is None:
                    additional_metadata = self.fetch_metadata_from_db(text, context_type)

                metadata.update(additional_metadata)
                print(f"Metadata: {metadata}")
                print(f"Text: {json.dumps(text)}")

                
                self.index.upsert([(vector_id, embedding_vector, metadata)])
        except Exception:
            print(f"An exception occurred: {e}")
            pass

    def fetch_metadata_from_db(self, text, context_type, interaction_id=None):
        db_conn = get_db_connection()
        cursor = None
        metadata = {}
        try:
            cursor = db_conn.cursor()
            
            if interaction_id:
                query = """SELECT session_id, conversation_id, user_id FROM conversation_history WHERE interaction_id = %s"""
                params = (interaction_id,)
                cursor.execute(query, params)
                result = cursor.fetchone()
                if result:
                    metadata.update(dict(result))

            if 'user_id' in metadata:
                query = """SELECT username, role_id, preferences, user_info FROM users WHERE user_id = %s"""
                params = (metadata['user_id'],)
                cursor.execute(query, params)
                result = cursor.fetchone()
                if result:
                    metadata.update(dict(result))

            if 'role_id' in metadata:
                query = """SELECT role_name FROM roles WHERE role_id = %s"""
                params = (metadata['role_id'],)
                cursor.execute(query, params)
                result = cursor.fetchone()
                if result:
                    metadata.update(dict(result))

            db_conn.commit()
            return metadata
        except Exception as e:
            if db_conn:
                db_conn.rollback()
            print(f"An error occurred while fetching metadata: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()

    def db_execute(self, query, params):
        db_conn = get_db_connection()
        cursor = None
        try:
            cursor = db_conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchall()
            db_conn.commit()
            return result
        except Exception as e:
            if db_conn:
                db_conn.rollback()
            print(f"An error occurred while executing the database query: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    def query(self, embedding_vector, top_k=5, threshold=0.9, context_filter=None):
        try:
            filter_query = {}
            if context_filter:
                filter_query["context_type"] = {"$eq": context_filter}

            if embedding_vector and self.index:
                results = self.index.query(
                    vector=embedding_vector, 
                    filter=filter_query,
                    top_k=top_k,
                    include_metadata=True  
                )

                results['matches'] = [match for match in results['matches'] if match['score'] >= threshold]
                
                return results
            else:
                return {'matches': []}
        except Exception as e:
            logger.error(f"Error in PineconeManager query method: {e}")
            return {'matches': []}

    def query_similar_embeddings(self, text: Union[str, List[str]], top_k=None, history_depth=5) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        if top_k is None:
            top_k = self.determine_dynamic_top_k(text)
        
        if isinstance(text, list) and isinstance(text[0], str):
            text = text[0]
        
        embedding_vector, _ = self.get_embedding(text)
        if embedding_vector is None:
            return []

        # Query Pinecone with the embedding to get top_k similar contexts
        results = self.query(embedding_vector, top_k=top_k)
        
        # Print metadata and score for each match (moved here, after `results` is defined)
        for match in results['matches']:
            print(f"Match metadata: {match['metadata']}")
            print(f"Match score: {match['score']}")
        
        # Here, we filter the results to only include the most relevant ones based on some criterion (e.g., a score threshold)
        similar_texts = [{"text": match['metadata']['original_text'], "metadata": match['metadata']} for match in results['matches'] if match['score'] > 0.8] 

        return similar_texts[:history_depth]

class PineconeMemory(BaseMemory):
    manager: PineconeManager

    def __init__(self, manager_instance: PineconeManager, **data):
        data['manager'] = manager_instance
        super().__init__(**data)

    @property
    def memory_variables(self) -> List[str]:
        return ["combined_memory"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            results = self.manager.query_similar_embeddings(inputs["input"])
            if results:
                for r in results:
                    for key, value in r.items():
                        if isinstance(value, datetime):
                            r[key] = value.isoformat()  # Convert datetime objects to strings
                merged_memory = ' '.join([json.dumps(r) for r in results])
                return {"memory": merged_memory}
        except Exception as e:
            # Log the exception (consider using a logger instead of print)
            print(f"Error in load_memory_variables: {e}")
        return {}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]):
        try:
            input_text = self._extract_text(inputs["input"])
            output_text = self._extract_text(outputs["output"])

            combined_text = {
                "input_text": input_text,
                "output_text": output_text
            }

            self.manager.save_embedding_to_pinecone(combined_text, "chat_response", additional_metadata={"type": "combined_text"})
        except Exception as e:
            # Log the exception (consider using a logger instead of print)
            print(f"Error in save_context: {e}")

    def _extract_text(self, data):
        if isinstance(data, list) and isinstance(data[0], HumanMessage):
            return " ".join([msg.content for msg in data])
        return data

    def clear(self):
        # Implement the logic to clear the memory variables or reset the memory space
        pass

# Usage:
pinecone_manager = PineconeManager()
pinecone_memory = PineconeMemory(pinecone_manager)

# Initialize Tools and Resources
llm = ChatOpenAI(temperature=0.2, model="gpt-4-0613")
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
db = get_db_connection()
engine = get_db_engine()
db_object = SQLDatabase(engine)
db_chain = SQLDatabaseChain.from_llm(llm, db_object, verbose=True)

tools = [
    Tool(name="Search", func=search.run, description="Answer questions about current events."),
    Tool(name="Calculator", func=llm_math_chain.run, description="Answer questions about math."),
    Tool(name="Postgres-DB", func=db_chain.run, description="Answer questions about Postgres.")
]


# Define the Agent
agent = initialize_agent(tools, llm, memory=pinecone_memory, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

# Instantiate the SystemMessageContent
system_message_content_instance = SystemMessageContent()

# Initialize the conversation with the system message
previous_messages = [SystemMessage(content=str(system_message_content_instance))]

print("\nAgent is now active. Type 'exit' to stop.")
history_depth = 5  # Consider the last 5 messages for context

session_start = True  # Flag to check if it is the start of the session


while True:
    try:
        # 1. Input Gathering and Validation
        user_input = input("User: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Initialize response variable
        response = None

        # File reading functionalities
        if user_input.lower().startswith('read file:'):
            file_path = user_input[len('read file:'):].strip()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                file_contents[file_path] = file_content  # Store file content in dictionary
                read_result = read_and_analyze_file(file_path)
            except Exception as e:
                read_result = str(e)
            user_input = read_result  # Replace user_input with the analysis result
        elif user_input.lower().startswith('access file content:'):
            file_path = user_input[len('access file content:'):].strip()
            file_content = file_contents.get(file_path, "File content not found")
            user_input = file_content  # Replace user_input with the file content

        # 3. Context Enrichment
        recent_input = user_input
        
        # Fetching similar texts and historical context from Pinecone
        similar_texts = pinecone_manager.query_similar_embeddings(user_input)
        historical_context = " | ".join([f'{item["text"]} (metadata: {item["metadata"]})' for item in similar_texts])
        
        recent_interactions = deque(maxlen=5)
        recent_interactions.append({"role": "user", "content": user_input, "timestamp": datetime.utcnow().isoformat()})
        
        # Generate recent interactions string
        recent_interactions_str = " | ".join([f'{item["role"]}: {item["content"]} (at {item["timestamp"]})' for item in recent_interactions])
        
        # Formulate the full input considering the session start
        if session_start:  # If it is the start of the session, include the system message
            full_input = f"System: {system_message_content_instance} | Recent: {recent_interactions_str} | Context: {historical_context}"
            session_start = False  # Reset the session start flag
        else:
            full_input = f"Recent: {recent_interactions_str} | Context: {historical_context}"
        
        # 4. Response Generation
        messages = [HumanMessage(content=full_input)]
        response_messages = agent.run(messages)
        response_content = response_messages[0].content
        response = response_content if isinstance(response_messages[0], HumanMessage) else response_messages[0]
        
        # Add the agent response to recent interactions and previous messages
        recent_interactions.append({"role": "agent", "content": response, "timestamp": datetime.utcnow().isoformat()})
        previous_messages.extend(recent_interactions)


        # 5. Logging and Storage
        try:
            pinecone_memory.save_context({"input": full_input}, {"output": response})
            # ... (continue with the rest of the logging and storage operations)
        except Exception as pinecone_error:
            logger.error(f"Error during Pinecone save: {pinecone_error}")

        # Get embedding vector and metadata
        embedding_vector, metadata = pinecone_manager.get_embedding(user_input)
                # Fetching the conversation history for context enrichment
        user_id = "cfd8c888-4273-4b4b-96ee-e7467642a840"

        # Record the user interaction in the PostgreSQL database
        interaction_info = {
            "user_input": user_input,
            "agent_response": response,
            "context": full_input,
            "embedding_vector_metadata": metadata
        }
        record_user_interaction(user_id, interaction_info)

        # 6. Output
        print("\nAgent:", response)

    except KeyboardInterrupt:
        # 7. Error Handling
        print("\nExiting the application. Have a great day!")
        break
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An error occurred: {e}. Please try again.")
        
# 8. Cleanup
db_conn = get_db_connection()
if db_conn:
    db_conn.close()
