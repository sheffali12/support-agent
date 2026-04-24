import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from contextlib import asynccontextmanager
from gtts import gTTS                          # NEW: text-to-speech
from langdetect import detect                  # NEW: language detection
import io
import uvicorn

load_dotenv()

# ── 1. RESPONSE SCHEMA ──────────────────────────────────────────────────────

class StructuredResponse(BaseModel):
    summary: str    = Field(description="A one-sentence summary of the answer.")
    details: List[str] = Field(description="2-5 detailed bullet points.")
    next_steps: List[str] = Field(description="Actionable next steps.")
    detected_lang: Optional[str] = Field(default="en", description="Detected language code: 'en' or 'hi'")

# ── 2. LANGUAGE DETECTION ───────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect if input is Hindi ('hi') or English ('en')."""
    try:
        lang = detect(text)
        return "hi" if lang == "hi" else "en"
    except Exception:
        return "en"

# ── 3. SUPPORT AGENT ────────────────────────────────────────────────────────

class SupportAgent:

    def __init__(self, pdf_path):
        self.pdf_path  = pdf_path
        self.store     = {}
        self.llm       = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.parser    = PydanticOutputParser(pydantic_object=StructuredResponse)
        self.vector_db = self._setup_knowledge_base()
        self.chain     = self._setup_chain()

    def _setup_knowledge_base(self):
        persist_dir = "./chroma_db"
        if os.path.exists(persist_dir):
            return Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        if not os.path.exists(self.pdf_path):
            print(f"Warning: {self.pdf_path} not found. Creating empty vector store.")
            return Chroma(embedding_function=self.embeddings, persist_directory=persist_dir)
        loader = PyPDFLoader(self.pdf_path)
        data   = loader.load()
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
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_prompt
        )

        # NEW: System prompt instructs LLM to respond in the same language as the user
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful banking support assistant for National Banking Services.

LANGUAGE RULE (CRITICAL):
- Detect the language of the user's question.
- If the question is in Hindi (हिंदी), respond ENTIRELY in Hindi — all fields (summary, details, next_steps) must be in Hindi.
- If the question is in English, respond entirely in English.
- Never mix languages within a single response.

Use the retrieved context to answer accurately.
You MUST follow the JSON format instructions strictly.

{format_instructions}

Context: {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]).partial(format_instructions=self.parser.get_format_instructions())

        qa_chain  = create_stuff_documents_chain(self.llm, qa_prompt)
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

    def ask(self, query: str, session_id: str = "default", lang_override: str = None):
        # Detect language — use override if frontend explicitly set one
        detected = lang_override if lang_override in ("en", "hi") else detect_language(query)
        try:
            result = self.chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
            parsed = self.parser.parse(result["answer"])
            data   = parsed.model_dump()
            data["detected_lang"] = detected   # attach to response
            return data
        except Exception as e:
            print(f"Logic Error: {e}")
            return {"error": "Failed to parse AI response. Please try again.", "detected_lang": detected}

# ── 4. FASTAPI APP ───────────────────────────────────────────────────────────

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

# ── 5. REQUEST MODELS ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str
    session_id: str = "default"
    lang:       Optional[str] = None   # NEW: optional manual override ("en" | "hi")

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"                   # "en" or "hi"

# ── 6. ENDPOINTS ─────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint — accepts question + optional lang override."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    response = agent.ask(request.question, request.session_id, request.lang)
    return {"data": response}


@app.post("/detect-language")
async def detect_lang_endpoint(payload: dict):
    """Utility: detect language of a given text string."""
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text field required")
    return {"detected_lang": detect_language(text)}


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    NEW: Text-to-Speech endpoint.
    Accepts { text, lang } → returns audio/mpeg stream.
    lang: "en" → English, "hi" → Hindi
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text field required")

    gtts_lang = "hi" if request.lang == "hi" else "en"

    try:
        tts = gTTS(text=request.text, lang=gtts_lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)