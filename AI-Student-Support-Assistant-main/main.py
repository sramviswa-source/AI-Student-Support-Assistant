from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Load the college knowledge base
# --------------------------------------------------

loader = TextLoader("college_info.txt")

documents = loader.load()

LLM_MODEL = "llama32-local"
EMBEDDINGS_MODEL = "nomic-embed-text"

# --------------------------------------------------
# 2. Split documents into smaller chunks
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Loaded {len(chunks)} document chunks")


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDINGS_MODEL
)


# --------------------------------------------------
# 4. Store embeddings in FAISS
# --------------------------------------------------

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# --------------------------------------------------
# 5. Create retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# --------------------------------------------------
# 6. Create Ollama LLM
# --------------------------------------------------

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)


# --------------------------------------------------
# 7. Create prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are an AI Student Support Assistant.

Answer the student's question using ONLY the provided college
information.

If the answer is not available in the provided information,
say:

"I don't have enough information in the college knowledge base."

Do not invent college policies.

College Information:
{context}

Student Question:
{question}

Answer clearly and concisely.
""")


# --------------------------------------------------
# 8. Ask a question
# --------------------------------------------------

def ask_question(question):

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Combine retrieved information
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Create final prompt
    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    # Ask Ollama
    response = llm.invoke(messages)

    return response.content


# --------------------------------------------------
# 9. Chat loop
# --------------------------------------------------

print("\nAI Student Support Assistant")
print("Type 'exit' to stop.\n")


while True:

    question = input("Student: ")

    if question.lower() == "exit":
        break

    answer = ask_question(question)

    print("\nAssistant:", answer)
    print()