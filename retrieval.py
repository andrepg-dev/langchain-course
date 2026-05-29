import os

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)


print("[*] Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Get the pinecone vector store data
"""Why do I need an embedding model to retrieve the data? what the hell"""
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME"), embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {question}

Provide a detailed answer:"""
)


# With this we get all the content information, I don't know why we will wanted this honestly, but that's okey.
# Maybe the model with provide us the type of chunks he needs and we need to give to him the page content of every chunk created.
def format_docs(docs):
    """format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query)

    print(f"The model get {len(docs)} chunks")

    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content


if __name__ == "__main__":
    print("[*] Retrieving...")

    query = "What is PineCone in Machine Learning?"

    # =============================================
    # Option 0: raw invocation without RAG
    # =============================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM invocation (No RAG)")
    response = llm.invoke([HumanMessage(content=query)])
    print(response.content)
    print("\n" + "=" * 70)

    # =============================================
    # Option 1: Using RAG without LangChain Expression Language (Sea lo que sea Expression Language)
    # =============================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Using RAG without LangChain Expression Language")
    print(retrieval_chain_without_lcel(query=query))
    print("\n" + "=" * 70)
