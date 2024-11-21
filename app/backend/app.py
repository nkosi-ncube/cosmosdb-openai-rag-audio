import os
import logging
from dotenv import load_dotenv
from aiohttp import web
from ragtools import attach_rag_tools
from rtmt import RTMiddleTier
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from urllib.parse import urlparse

# Set up basic logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    load_dotenv()
    
    # Load environment variables
    llm_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    llm_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")
    mongo_connection_string = os.environ.get("MONGO_CONNECTION_STRING")
    database_name = os.environ.get("MONGO_DB_NAME")
    collection_name = os.environ.get("MONGO_COLLECTION_NAME")
    
    if not all([llm_endpoint, llm_deployment, mongo_connection_string, database_name, collection_name]):
        logging.error("One or more required environment variables are missing.")
        exit(1)

    logging.info(f"DATABASE NAME: {database_name}")
    logging.info(f"COLLECTION_NAME: {collection_name}")

    # Extract the base URL from the LLM endpoint
    parsed_url = urlparse(llm_endpoint)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    logging.info(f"Base URL extracted: {base_url}")

    # Credentials for Azure
    credentials = DefaultAzureCredential() if not llm_key else None

    app = web.Application()

    # Initialize the real-time middleware with Azure OpenAI
    try:
        rtmt = RTMiddleTier(
            base_url,  # Use the base URL instead of the full endpoint
            llm_deployment, 
            AzureKeyCredential(llm_key) if llm_key else credentials
        )
        rtmt.system_message = (
            "You are a helpful assistant. Only answer questions based on information you searched in the knowledge base, "
            "accessible with the 'search' tool. "
            "The user is listening to answers with audio, so it's *super* important that answers are as short as possible, "
            "a single sentence if at all possible. "
            "Never read file names or source names or keys out loud. "
            "Always use the following step-by-step instructions to respond: \n"
            "1. Always use the 'search' tool to check the knowledge base before answering a question. \n"
            "2. Always use the 'report_grounding' tool to report the source of information from the knowledge base. \n"
            "3. Produce an answer that's as short as possible. If the answer isn't in the knowledge base, say you don't know."
        )
    except Exception as e:
        logging.error(f"Error initializing RTMiddleTier: {e}")
        exit(1)
    
    pdf_dir = "../../data"

    # Attach MongoDB-based vector search to the real-time middleware
    try:
        attach_rag_tools(rtmt, mongo_connection_string, database_name, collection_name, pdf_dir)
    except Exception as e:
        logging.error(f"Error attaching RAG tools: {e}")
        exit(1)

    # Attach to app
    rtmt.attach_to_app(app, "/realtime")

    # Serve index.html and static files
    app.add_routes([web.get('/', lambda _: web.FileResponse('./static/index.html'))])
    app.router.add_static('/', path='./static', name='static')

    try:
        logging.info("Starting the server on http://localhost:8765")
        web.run_app(app, host='localhost', port=8765)
    except Exception as e:
        logging.error(f"Error running the web app: {e}")
