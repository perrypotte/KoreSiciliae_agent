import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_postgres import PGVector
import getpass
import os
os.environ["NVIDIA_API_KEY"] = "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"
if not os.environ.get("NVIDIA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter API key for NVIDIA: ")

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="embeddings",
    connection="postgresql://postgres:postgres@localhost:5432/mydb",
)


""" bs4_strainer = bs4.SoupStrainer()
loader = WebBaseLoader(
    web_paths=("https://www.koresiciliae.it/ci-presentiamo",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
 """
from bs4 import BeautifulSoup
loader = WebBaseLoader(
    web_paths=("https://www.koresiciliae.it/ci-presentiamo",),
)

docs = loader.load()

html = docs[0].page_content

soup = BeautifulSoup(html, "html.parser")
# Rimuovi roba inutile
for tag in soup(["script", "style", "nav", "footer", "header"]):
    tag.decompose()
text = text = soup.get_text(separator="\n", strip=True)
lines = text.split("\n")
clean_lines = [line for line in lines if len(line) > 40]  # tieni solo frasi "vere"
clean_text = "\n".join(clean_lines)

print(clean_text)
""" print(text) """

assert len(docs) == 1

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,  # chunk size (characters)
    chunk_overlap=50,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
#print(f"Total characters: {len(docs[0].page_content)}")
#print(docs[0].page_content[:500])