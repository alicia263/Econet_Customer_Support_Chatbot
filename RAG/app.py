import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv

# Import the RAG functions
from rag import rag, submit_feedback
from db import get_db_connection

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

# Pydantic models for request bodies
class QuestionRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    conversation_id: str
    feedback: int

@app.on_event("startup")
async def startup_event():
    # Check database connection on startup
    conn = get_db_connection()
    if conn is None:
        logging.error("Failed to connect to the database. Please check your database configuration.")
    else:
        conn.close()
        logging.info("Successfully connected to the database.")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        logging.info(f"Received question: {request.question}")
        
        # Generate answer using RAG
        answer_data = rag(request.question)
        
        logging.info(f"Generated answer for conversation ID: {answer_data['id']}")
        
        return {
            "conversation_id": answer_data['id'],
            "answer": answer_data['answer'],
            "relevance": answer_data['relevance'],
            "response_time": answer_data['response_time'],
            "total_tokens": answer_data['total_tokens'],
            "estimated_cost": answer_data['openai_cost']
        }
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your question")

@app.post("/feedback")
async def process_feedback(request: FeedbackRequest):
    try:
        logging.info(f"Received feedback for conversation ID: {request.conversation_id}")
        
        if request.feedback not in [-1, 1]:
            raise HTTPException(status_code=400, detail="Invalid feedback value. Must be -1 or 1")
        
        # Submit feedback to the database
        result = submit_feedback(request.conversation_id, request.feedback)
        
        if result['status'] == 'success':
            return {"message": "Feedback received and saved successfully"}
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your feedback")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)