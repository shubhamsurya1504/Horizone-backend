from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import AsyncGenerator, List
import sqlite3
import numpy as np

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MYINDEX = os.getenv("MYINDEX")
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Ensure PINECONE_API_KEY and OPENAI_API_KEY are set in the environment.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name =  MYINDEX
index = pc.Index(name=index_name)

# Function to resize embeddings to match Pinecone index dimensions
def resize_embedding(embedding, target_size=1024):
    """Resize embedding to target size"""
    if len(embedding) == target_size:
        return embedding
    
    # Convert to numpy array
    embedding_array = np.array(embedding)
    
    if len(embedding) > target_size:
        # Downsample (take first target_size elements)
        return embedding_array[:target_size].tolist()
    else:
        # Upsample (pad with zeros)
        padded = np.zeros(target_size)
        padded[:len(embedding)] = embedding_array
        return padded.tolist()

# Initialize OpenAI model and custom embeddings
class ResizedOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        embeddings = super().embed_documents(texts)
        return [resize_embedding(emb) for emb in embeddings]

    def embed_query(self, text):
        embedding = super().embed_query(text)
        return resize_embedding(embedding)

# Initialize models with dimension handling
embedding_model = ResizedOpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = OpenAI(temperature=0,api_key=OPENAI_API_KEY)

# Create custom prompt template
prompt_template = """
You are an AI assistant specialized in Horizon Europe grant writing, focusing on Research and Innovation Actions (RIAs) and Innovation Actions (IAs). Your role is to generate high-quality, structured responses using a **retrieval-augmented approach** based on the provided document context.

### **System Instructions:**
- Begin your response by providing the **whole call topic description** including:
  - Action type (RIA, IA or CSA)
  - Funding amount per action
  - Expected outcomes
  - Scope
  - Expected Impacts
- Retrieve the most relevant information from the provided context.
- Structure responses according to Horizon Europe's official proposal sections.
- If information is missing or unclear, acknowledge the limitation and suggest best practices.
- Use **bullet points, markdown formatting, and quantifiable indicators** where applicable.
- Cite specific sections or page numbers from the documents when possible.

### **Proposal Structure & Requirements:**
#### **1. Excellence**  
- Clearly define the project's **objectives and ambition**.  
- Address **state-of-the-art research, gaps, and innovation potential**.  
- Outline the **methodology**, including interdisciplinary, SSH, gender, and Open Science aspects.  
- Define the **TRL level** and explain how the project advances technological readiness.  

#### **2. Impact**  
- Develop a structured **impact pathway** linking expected outcomes to **societal, scientific, and economic benefits**.  
- Identify and **quantify key impact indicators** (e.g., CO2 reduction, jobs created, technology adoption).  
- Provide a strong **Dissemination, Exploitation, and Communication (DEC) plan**, detailing target groups and engagement strategies.  

#### **3. Implementation**  
- Structure a **work plan** with clear **work packages (WPs), deliverables, milestones, and effort distribution**.  
- Include **Gantt & PERT charts**, risk management strategies, and consortium capacity details.  
- Justify the budget and partner roles, ensuring alignment with Horizon Europe's funding expectations.  

---

### **Context:**  
{context}  

### **Question:**  
{question}  

### **Answer:**  
"""


PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create Pinecone vector store
vector_store = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")
retriever = vector_store.as_retriever()

# Create QA chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# SQLite database setup for chat history
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT
    )
""")
conn.commit()

# Request Model
class QueryRequest(BaseModel):
    query: str

async def stream_answer(query: str) -> AsyncGenerator[str, None]:
    """Streams the response in real-time."""
    print(f"Retrieving context from Pinecone for query: '{query}'")
    
    # Get documents from retriever directly to print them
    retrieved_docs = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(retrieved_docs)} documents from Pinecone")
    print("\n===== FULL CONTEXT FROM PINECONE =====\n")
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i+1}:")
        print(f"Content (FULL):\n{doc.page_content}")
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"Metadata: {doc.metadata}")
        print("=" * 80)
    print("\n===== END OF RETRIEVED CONTEXT =====\n")
    
    response = await asyncio.to_thread(qa_chain.invoke, {"query": query})
    answer = response['result']

    # Store in SQLite
    cursor.execute("INSERT INTO chat_history (question, answer) VALUES (?, ?)", (query, answer))
    conn.commit()

    for word in answer.split():
        yield word + " "
        await asyncio.sleep(0.05)  # Simulating streaming delay

@app.post("/query/")
async def get_answer(request: QueryRequest):
    """Retrieves the best-matching answer from the PDF-stored embeddings."""
    try:
        return StreamingResponse(stream_answer(request.query), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/")
def get_chat_history() -> List[dict]:
    """Returns the chat history."""
    cursor.execute("SELECT question, answer FROM chat_history ORDER BY id DESC")
    history = [{"question": row[0], "answer": row[1]} for row in cursor.fetchall()]
    return history

@app.delete("/clear_history/")
def clear_chat_history():
    """Deletes all chat history entries."""
    try:
        cursor.execute("DELETE FROM chat_history")
        conn.commit()
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")
