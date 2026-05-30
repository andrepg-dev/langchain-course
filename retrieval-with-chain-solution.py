from operator import itemgetter
import os

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console

console = Console()

load_dotenv(override=True)


print("[*] Initializing components...")

embeddings = OpenAIEmbeddings()
# llm = ChatOllama(model="qwen3:1.7b")
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


def create_retrieval_chain_with_lcel():
    # This takes the user question, and execute the retriever with the question, and then format_docs is executed.
    # This means: make context from question.
    context = itemgetter("question") | retriever | format_docs

    # This is the key
    # console.print(context.invoke({"question": "what is pinecone"}))

    runnable_passthrough_assign = RunnablePassthrough.assign(context=context)
    
    retrieval_chain = (
        {"question": itemgetter("question"), "context": context}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("[*] Retrieving...")

    query = "What is PineCone in Machine Learning?"

    # =============================================
    # Option 0: raw invocation without RAG
    # =============================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM invocation (No RAG)")
    parser = StrOutputParser()
    response = llm.stream([HumanMessage(content=query)])
    for chunk in parser.transform(response):
        print(chunk, flush=True, end="")
    print("\n" + "=" * 70)

    # =============================================
    # Option 1: Using RAG with LangChain Expression Language (Sea lo que sea Expression Language)
    # =============================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Using LangChain Expression Language")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.stream({"question": query})
    print("\nAnswer:")
    for chunk in result_with_lcel:
        print(chunk, flush=True, end="")
    print("\n" + "=" * 70)
