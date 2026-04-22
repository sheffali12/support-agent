import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
# Fixed import for standard chains
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from contextlib import asynccontextmanager
import uvicorn

load_dotenv()

# --- 1. RESPONSE SCHEMA ---
class StructuredResponse(BaseModel):
    summary: str = Field(description="A one-sentence summary of the answer.")
    details: List[str] = Field(description="2-5 detailed bullet points.")
    next_steps: List[str] = Field(description="Actionable next steps.")

# --- 2. CORE LOGIC ---
class SupportAgent:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.store = {}
        # Lower temperature to 0.0 for stricter formatting adherence
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.parser = PydanticOutputParser(pydantic_object=StructuredResponse)
        self.vector_db = self._setup_knowledge_base()
        self.chain = self._setup_chain()

    def _setup_knowledge_base(self):
        persist_dir = "./chroma_db"
        if os.path.exists(persist_dir):
            return Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        
        loader = PyPDFLoader(self.pdf_path)
        data = loader.load()
        # Slightly smaller chunks for more precise retrieval
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_documents(data)
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )

    def _setup_chain(self):
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history, reformulate the user question to be standalone."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_prompt)

        # Added format_instructions to the prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful support assistant. Use the provided context to answer the user's question.
            
            {format_instructions}
            
            If the answer is not in the context, use the following:
            Summary: I'm sorry, I don't have that information.
            Details: ["This topic isn't in my current database."]
            Next Steps: ["Please contact human support."]

            Context: {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]).partial(format_instructions=self.parser.get_format_instructions())

        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def ask(self, query, session_id="default"):
        try:
            result = self.chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
            # The 'answer' is now a string that needs to be parsed into our Pydantic model
            return self.parser.parse(result["answer"])
        except Exception as e:
            return {"error": str(e)}

# --- 3. FASTAPI ---
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = SupportAgent("company_handbook.pdf")
    yield

app = FastAPI(title="AI Support Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/chat")
async def chat(request: ChatRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # This now returns a dictionary/JSON object instead of just a string
    response = agent.ask(request.question, request.session_id)
    return {"data": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)