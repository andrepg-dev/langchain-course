import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from rich.console import Console

console = Console()

load_dotenv(override=True)

if __name__ == "__main__":
    loader = TextLoader("./mediumblog1.txt", encoding="UTF-8")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_splitted = text_splitter.split_documents(document)

    print(f"created ({len(text_splitted)}) chunks")

    embeddings = OpenAIEmbeddings()

    print("[*] Ingesting...")
    PineconeVectorStore.from_documents(
        text_splitted, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"]
    )

    print("[*] Finished.")
