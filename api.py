import time

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from auth import router as auth_router
from auth import get_current_user

from run_graph import app as rag_app


# ============================================================
# FASTAPI APPLICATION
# ============================================================

api = FastAPI(
    title="EnterpriseMind AI API",
    version="1.0.0",
    description="Secure Agentic RAG API with JWT Authentication and RBAC",
)


# ============================================================
# CORS
# ============================================================

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

api.include_router(auth_router)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    question: str


# ============================================================
# ROOT
# ============================================================

@api.get("/")
def root():
    return {
        "message": "EnterpriseMind AI API is running"
    }


# ============================================================
# CURRENT USER
# ============================================================

@api.get("/me")
async def get_me(
    user: dict = Depends(get_current_user)
):
    return {
        "username": user["username"],
        "role": user["role"],
    }


# ============================================================
# ASK
# ============================================================

@api.post("/ask")
async def ask_question(
    request: AskRequest,
    user: dict = Depends(get_current_user),
):
    # Start latency measurement
    start_time = time.perf_counter()

    # Role comes from the authenticated backend identity.
    # It is NOT taken from the frontend request.
    user_role = user["role"]

    initial_state = {
        "question": request.question,
        "user_role": user_role,
    }

    # Run the Agentic RAG workflow
    result = rag_app.invoke(initial_state)

    # End latency measurement
    end_time = time.perf_counter()
    latency = end_time - start_time

    return {
        "question": request.question,
        "username": user["username"],
        "user_role": user_role,
        "answer": result.get("final_answer"),
        "security_status": result.get("security_status"),
        "verifier_verdict": result.get("verification_status"),
        "citations": result.get("citations", []),
        "latency_seconds": round(latency, 2),
        "latency_breakdown": result.get("latency", {}),
    }	