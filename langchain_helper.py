import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()  # Load API key from .env file

# Get Gemini API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Function to call Gemini API for text generation
def call_gemini_api(prompt):
    # Use the Google Generative AI SDK to configure the model
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")  # Specify the Gemini model
    response = model.generate_content(prompt)  # Removed 'temperature' argument

    # Check if response is valid and return the generated content
    if response:
        return response.text
    else:
        return "I don't know."

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet (replace with the path to your actual CSV file)
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever()

    prompt_template = """Given the following context and a question, generate a short and creative answer.
    Keep it concise, but add a touch of flair or extra detail to make the answer stand out, while staying relevant to the context.

    CONTEXT: {context}

    QUESTION: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the QA chain
    def qa_chain(query):
        # Use the retriever to get the context from the vector database
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "I don't know."

        # Get the context from the retrieved documents
        context = docs[0].page_content  # Assuming you want the first document's content

        # Format the prompt for Gemini API
        prompt = PROMPT.format(context=context, question=query)

        # Generate the answer using Gemini API
        answer = call_gemini_api(prompt)
        if answer:
            return answer
        else:
            return "I don't know."

    return qa_chain

if __name__ == "__main__":
    create_vector_db()  # Create and save the vector database
    chain = get_qa_chain()  # Get the QA chain

    # Test the QA chain with a query
    query = "Do you have javascript course?"
    answer = chain(query)
    print(answer)  # Output the answer to the console
