from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import dotenv

# Load environment variables (if any)
dotenv.load_dotenv()

# Function to query the vector store
def query_vector_store(query, vector_store_path='vectorstore.faiss'):
    # Load the pre-existing FAISS vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        folder_path=vector_store_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()
    
    # Initialize the language model (OpenAI)
    llm = OpenAI()
    
    # Set up the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    
    # Run the query through the chain and get the result
    result = chain.invoke(query)
    return result

# Main loop for querying
if __name__ == "__main__":
    while True:
        # Get the query input from the user
        query = input("Enter your query (or type 'exit' to quit): ")
        
        # Exit the loop if the user types 'exit'
        if query.lower() == 'exit':
            break
        
        # Query the vector store and print the result
        result = query_vector_store(query)
        print(result)
