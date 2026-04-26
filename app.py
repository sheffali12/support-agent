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
from gtts import gTTS                          # text-to-speech
from langdetect import detect                  # language detection
import io
import uvicorn

load_dotenv()

# ── 1. RESPONSE SCHEMA 

class StructuredResponse(BaseModel):
    summary: str       = Field(description="A one-sentence summary of the answer.")
    details: List[str] = Field(description="2-5 detailed bullet points.")
    next_steps: List[str] = Field(description="Actionable next steps.")
    detected_lang: Optional[str] = Field(default="en", description="Detected language code: 'en' or 'hi'")
    sentiment: Optional[str] = Field(default="neutral", description="Sentiment of user message: 'positive', 'negative', or 'neutral'")
    empathy_note: Optional[str] = Field(default="", description="A short empathetic note shown only when sentiment is negative or frustrated.")

# ── 2. LANGUAGE DETECTION 

def detect_language(text: str) -> str:
    """Detect if input is Hindi ('hi') or English ('en')."""
    try:
        lang = detect(text)
        return "hi" if lang == "hi" else "en"
    except Exception:
        return "en"

# ── 3. SENTIMENT DETECTION

# Keyword-based fast sentiment check (no extra library needed)
NEGATIVE_KEYWORDS_EN = [
    "frustrated", "angry", "annoyed", "upset", "terrible", "horrible", "worst",
    "useless", "pathetic", "hate", "disgusting", "unacceptable", "ridiculous",
    "scam", "fraud", "cheating", "robbery", "stolen", "lost", "problem",
    "issue", "complaint", "not working", "broken", "failed", "error",
    "disappointed", "helpless", "stuck", "urgent", "immediately", "asap"
]
NEGATIVE_KEYWORDS_HI = [
    "गुस्सा", "परेशान", "निराश", "बेकार", "खराब", "समस्या", "शिकायत",
    "धोखा", "फ्रॉड", "चोरी", "तुरंत", "जरूरी", "नहीं हो रहा", "काम नहीं",
    "बंद हो गया", "पैसे गए", "नुकसान"
]
POSITIVE_KEYWORDS_EN = [
    "thank", "thanks", "great", "excellent", "wonderful", "perfect",
    "helpful", "amazing", "love", "happy", "satisfied", "good", "nice"
]
POSITIVE_KEYWORDS_HI = [
    "धन्यवाद", "शुक्रिया", "बढ़िया", "अच्छा", "खुश", "संतुष्ट", "मदद मिली"
]

def detect_sentiment(text: str) -> str:
    """Returns 'positive', 'negative', or 'neutral'."""
    lo = text.lower()
    for kw in NEGATIVE_KEYWORDS_EN + NEGATIVE_KEYWORDS_HI:
        if kw in lo:
            return "negative"
    for kw in POSITIVE_KEYWORDS_EN + POSITIVE_KEYWORDS_HI:
        if kw in lo:
            return "positive"
    return "neutral"

EMPATHY_MESSAGES = {
    "en": "I understand this may be frustrating. I'm here to help and will do my best to resolve this for you.",
    "hi": "मैं समझता हूँ कि यह परेशान करने वाला हो सकता है। मैं आपकी पूरी सहायता करने के लिए यहाँ हूँ।"
}

# ── 3. SUPPORT AGENT 

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

        # System prompt instructs LLM to respond in user's language + show empathy if negative
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful banking support assistant for National Banking Services.

LANGUAGE RULE (CRITICAL):
- Detect the language of the user's question.
- If the question is in Hindi (हिंदी), respond ENTIRELY in Hindi — all fields (summary, details, next_steps, empathy_note) must be in Hindi.
- If the question is in English, respond entirely in English.
- Never mix languages within a single response.

SENTIMENT RULE:
- If the user seems frustrated, angry, or upset, set sentiment to "negative" and write a short empathetic empathy_note.
- If the user seems happy or thankful, set sentiment to "positive" and leave empathy_note empty.
- Otherwise set sentiment to "neutral" and leave empathy_note empty.

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
        detected  = lang_override if lang_override in ("en", "hi") else detect_language(query)
        sentiment = detect_sentiment(query)
        try:
            result = self.chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
            parsed = self.parser.parse(result["answer"])
            data   = parsed.model_dump()
            data["detected_lang"] = detected
            # If keyword sentiment says negative but LLM didn't catch it, ensure empathy
            if sentiment == "negative" and not data.get("empathy_note"):
                data["empathy_note"] = EMPATHY_MESSAGES.get(detected, EMPATHY_MESSAGES["en"])
            data["sentiment"] = sentiment if sentiment != "neutral" else data.get("sentiment", "neutral")
            return data
        except Exception as e:
            print(f"Logic Error: {e}")
            return {"error": "Failed to parse AI response. Please try again.", "detected_lang": detected, "sentiment": "neutral"}

# ── 4. FASTAPI APP 

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

# ── 5. REQUEST MODELS 

class ChatRequest(BaseModel):
    question:   str
    session_id: str = "default"
    lang:       Optional[str] = None   # NEW: optional manual override ("en" | "hi")

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"                   # "en" or "hi"

# ── 6. ENDPOINTS

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


# ── 7. EMI CALCULATOR 

class EMIRequest(BaseModel):
    principal:     float = Field(description="Loan amount in rupees")
    annual_rate:   float = Field(description="Annual interest rate in percent (e.g. 8.5)")
    tenure_years:  float = Field(description="Loan tenure in years (e.g. 5)")

@app.post("/calculate-emi")
async def calculate_emi(request: EMIRequest):
    """
    EMI Calculator — reducing balance method.
    Returns monthly EMI, total interest, total amount, and a yearly breakdown.
    """
    if request.principal <= 0 or request.annual_rate <= 0 or request.tenure_years <= 0:
        raise HTTPException(status_code=400, detail="All values must be positive")

    P = request.principal
    r = request.annual_rate / 12 / 100        
    n = int(request.tenure_years * 12)        

    # Standard EMI formula (reducing balance)
    emi = P * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total_amount   = emi * n
    total_interest = total_amount - P

    # Year-by-year breakdown
    breakdown = []
    balance = P
    for year in range(1, int(request.tenure_years) + 1):
        year_principal = 0
        year_interest  = 0
        months = min(12, n - (year - 1) * 12)
        for _ in range(months):
            interest   = balance * r
            principal  = emi - interest
            balance   -= principal
            year_principal += principal
            year_interest  += interest
        breakdown.append({
            "year":           year,
            "principal_paid": round(year_principal, 2),
            "interest_paid":  round(year_interest,  2),
            "balance":        round(max(balance, 0), 2)
        })

    return {
        "emi":            round(emi, 2),
        "total_amount":   round(total_amount, 2),
        "total_interest": round(total_interest, 2),
        "principal":      round(P, 2),
        "annual_rate":    request.annual_rate,
        "tenure_years":   request.tenure_years,
        "tenure_months":  n,
        "breakdown":      breakdown,
        "disclaimer":     "For illustration purposes only. Actual EMI may vary based on bank policies."
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)