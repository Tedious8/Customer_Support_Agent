# Support Agent — Multi-Step AI Customer Support System

A production-ready multi-step AI agent for customer support, built with
**LangGraph**, **FastAPI**, and the **Gemini API**.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Start the API server

```bash
python main.py
# or: uvicorn main:app --reload
```

---

## API Endpoints

| Method | Path      | Description  |
| ------ | --------- | ------------ |
| GET    | `/health` | Health check |

---

## File Structure

```
support_agent/
├── main.py            # FastAPI app
├── requirements.txt
├── .env.example
└── README.md
```
