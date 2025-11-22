🤖 INTELEXI — Ask. Understand. Intelexi.
![Intellexi Sticker](/assets/A_digital_illustration_features_a_sticker-style_lo.png)
Intelexi is an intelligent, multi-mode document assistant built using Streamlit, powered by Groq LLaMA-3 models, and enhanced with semantic search, PDF understanding, and speech-to-text capabilities.

It provides highly accurate responses based on:
PDF content
Voice queries
Wikipedia fallback
AI reasoning
All within a clean, modern, responsive UI.

✨ Key Features
📄 1. Document-Based Q&A
        1.Upload one or more PDFs
        2.Intelexi extracts text using PyPDF2
        3.Splits content into chunks using semantic text splitters
        4.Embeds them using SentenceTransformers
        5.Retrieves the most relevant content for your question

🎤 2. Voice Input Processing
        1.Ask questions through audio files
        2.Faster-Whisper transcribes speech with high accuracy
        3.The transcription is passed directly to the LLaMA model

🌐 3. Wikipedia Fallback
        If a question cannot be answered from your documents, Intelexi automatically queries Wikipedia and synthesizes a helpful response.

🧠 4. AI Reasoning Fallback
        If the document and Wikipedia both fail, Intelexi uses Groq LLaMA-3.1-8B Instant for fast, intelligent responses.

💬 5. Chat History
        All interactions appear as beautifully styled chat bubbles — for both user and AI messages.

🎨 6. Custom UI Styling
Intelexi features:
        1.Gradient background
        2.Soft geometric textures
        3.Clean side navigation    
        4.Modern chat UI with readable spacing
        5.Icons, colors, and a polished visual experience
        
🛠️ Tech Stack

🛠️ Tech Stack
Component	Technology
User Interface	: Streamlit
LLM Backend	: Groq LLaMA-3.1-8B Instant
Speech-to-Text	: Whisper
PDF Parsing	: PyPDF2
Embeddings	: SentenceTransformers
Semantic Search	: NumPy cosine similarity
External Knowledge	: Wikipedia API
UI	: Custom CSS + HTML styling
![Intelexi logo](assets/intelexi_logo.png)
![Intelexi UI](assets/home_screen.png)

Architecture

                            ┌─────────────────────────┐
                            │        User Input        │
                            │  • PDF Uploads           │
                            │  • Text Questions        │
                            │  • Voice Questions       │
                            └────────────┬────────────┘
                                         │
                                         ▼
                        ┌──────────────────────────────────┐
                        │         Mode Controller          │
                        │  Streamlit (Home / Text / Voice) │
                        └────────────┬─────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                         PDF Processing                          │
        │  • PyPDF2 → Extract raw text                                    │
        │  • Clean + normalize text                                       │
        │  • Chunk text (200 words)                                       │
        │  • SentenceTransformers → Generate embeddings                   │
        │  • NumPy → Vector similarity search (Top-K chunk retrieval)     │
        └───────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  Is answer found?  │
                     └───────┬────────────┘
                             │  YES
                             ▼
                     ┌────────────────────┐
                     │  LLaMA Answering   │
                     │ Groq API (8B model)│
                     └─────────┬──────────┘
                               │
                               ▼
                         FINAL ANSWER
                               ▲
                               │ NO
                               │
                               ▼
         ┌─────────────────────────────── Wikipedia Fallback ───────────────────────────────┐
         │  • wikipedia.summary()                                                           │
         │  • If successful → Send combined prompt to LLaMA                                │
         │  • If fails → Skip to LLaMA model fallback                                       │
         └───────────────────────────────┬──────────────────────────────────────────────────┘
                                         │
                                         ▼
                       ┌─────────────────────────────────────┐
                       │     LLaMA Reasoning Fallback        │
                       │ (General knowledge, coding, logic)  │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                                   FINAL ANSWER


