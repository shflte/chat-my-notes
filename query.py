from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain import hub
import dotenv


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query_vector_store(query, vector_store_path="vectorstore.faiss"):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        folder_path=vector_store_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-4o")

    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(query)
    return result


dotenv.load_dotenv()
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    result = query_vector_store(query)
    print(result)
