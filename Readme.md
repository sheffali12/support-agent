# 🏦 NBS — AI Customer Support Agent

> An LLM-powered, voice-enabled, bilingual banking assistant built with RAG architecture.

**🔗 Live Demo:** https://sheffali12.github.io/support-agent  
**📡 API Docs:** https://support-agent-production-ce11.up.railway.app/docs

---

## 📸 Preview

![NBS Banking Assistant](https://sheffali12.github.io/support-agent/preview.png)

---

## 🎯 What It Does

NBS is an intelligent customer support chatbot for National Banking Services. It answers domain-specific banking queries using a RAG (Retrieval-Augmented Generation) pipeline, understands conversation context, detects the user's language and sentiment, and responds with empathy — all in real time.

---

## ✨ Features

| Feature | Description |

| 🧠 **RAG Architecture** | Retrieves answers from a banking knowledge base (PDF) using ChromaDB vector search |
| 💬 **Multi-turn Memory** | Remembers conversation context within a session |
| 🌐 **Multi-language** | Auto-detects Hindi/English and responds in the same language. Manual override available |
| 🎙️ **Voice Input** | Speak your question using the mic — supports both English and Hindi |
| 🔊 **Voice Output** | Every response can be read aloud via Google TTS in the correct language |
| ❤️ **Sentiment Detection** | Detects frustrated/negative messages — shows empathy note and highlights the response in red |
| 📋 **Structured Responses** | Every answer includes a Summary, Details, and Next Steps |
| ⚡ **Fast** | Powered by Llama 3.1 8B via Groq — responses in under 2 seconds |

---

## 🏗️ Architecture

```
User Message
     │
     ▼
Frontend (GitHub Pages)
     │  POST /chat
     ▼
FastAPI Backend (Railway)
     │
     ├── detect_language()  →  "en" or "hi"
     ├── detect_sentiment()  →  "positive" / "negative" / "neutral"
     │
     ▼
LangChain RAG Pipeline
     │
     ├── History-aware retriever  →  ChromaDB (company_handbook.pdf)
     ├── Llama 3.1 8B via Groq
     └── Pydantic output parser  →  { summary, details, next_steps, sentiment, empathy_note }
     │
     ▼
Structured JSON Response
     │
     ▼
Frontend renders color-coded bubble + empathy note
```

---

## 🛠️ Tech Stack

**Backend**
- FastAPI + Uvicorn
- LangChain + LangChain-Groq
- Llama 3.1 8B (via Groq)
- ChromaDB (vector store)
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- gTTS (Google Text-to-Speech)
- langdetect (language detection)
- PyPDF
- Pydantic

**Frontend**
- Vanilla HTML/CSS/JS
- Web Speech API (voice input)
- Marked.js (markdown rendering)

**Infrastructure**
- Railway (backend hosting)
- GitHub Pages (frontend hosting)
- Docker

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Main chat endpoint |
| `POST` | `/tts` | Text-to-speech — returns MP3 audio |
| `POST` | `/detect-language` | Detect language of a text string |
| `GET` | `/docs` | Swagger UI |

### `/chat` Request
```json
{
  "question": "How do I apply for a loan?",
  "session_id": "user_abc123",
  "lang": null
}
```

### `/chat` Response
```json
{
  "data": {
    "summary": "You can apply for a loan online or at any branch.",
    "details": ["...", "..."],
    "next_steps": ["...", "..."],
    "detected_lang": "en",
    "sentiment": "neutral",
    "empathy_note": ""
  }
}
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/sheffali12/support-agent.git
cd support-agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set environment variables**
```bash
# Create a .env file
GROQ_API_KEY=your_groq_api_key_here
```

**4. Add your knowledge base**

Place your `company_handbook.pdf` in the root directory.

**5. Run the server**
```bash
python app.py
```

Server starts at `http://localhost:8000`

**6. Open the frontend**

Open `index.html` in your browser or serve it locally:
```bash
npx serve .
```

---

## 🧪 Demo Script

Try these messages to see all features in action:

| Message | Expected behaviour |
|---|---|
| `How do I open a savings account?` | Normal structured response |
| `मेरा खाता कैसे खोलें?` | Auto-switches to Hindi, replies in Hindi |
| `I am very frustrated, my transfer failed` | Red bubble + ⚠️ URGENT badge + empathy note |
| `Thank you, that was very helpful!` | Green bubble + 😊 Positive badge |
| *(click mic and speak)* | Voice transcription + response |
| *(click Listen on any response)* | TTS audio playback |

---

## 📁 Project Structure

```
support-agent/
├── app.py                  # FastAPI backend
├── index.html              # Frontend UI
├── requirements.txt        # Python dependencies
├── company_handbook.pdf    # Banking knowledge base
├── Dockerfile              # Container config
├── chroma_db/              # Vector store (auto-generated)
└── .env                    # API keys (not committed)
```

---

## 🙋 Problem Statement

**Statement 5 — AI Customer Support Agent (Next-Gen Chatbot)**

> Develop an LLM-powered conversational AI assistant for customer support using LLMs + RAG architecture capable of answering domain-specific queries, understanding context, accessing knowledge bases, and providing human-like responses.

**Advanced features implemented:** Multi-language support · Voice chatbot · Sentiment detection


GitHub: [@sheffali12](https://github.com/sheffali12)