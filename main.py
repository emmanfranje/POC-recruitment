import io
import os
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from markitdown import MarkItDown
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(
    title="Automated AI Interviewer Core API",
    description="Phase 2: Stateful Conversational Audio Screening & Dynamic Evaluation",
    version="2.0.0"
)

# Enable CORS so our frontend index.html can communicate with our local server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

md_converter = MarkItDown()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is missing from your environment setup.")
ai_client = genai.Client()

# --- In-Memory Session Storage ---
# For our MVP, we store active interviews in memory.
# structure: { session_id: { "state_data": ... } }
active_sessions: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Schemas ---

class ScreeningQuestion(BaseModel):
    id: int = Field(..., description="Sequential index of the question (1-5).")
    topic: str = Field(..., description="The high-level technology or requirement being evaluated.")
    question_text: str = Field(..., description="The conversational interview question.")
    evaluation_criteria: str = Field(..., description="What constitutes a strong vs shallow answer.")

class InterviewInitializationResponse(BaseModel):
    session_id: str = Field(..., description="A unique UUID assigned to this candidate session.")
    candidate_name: str = Field(..., description="The extracted name of the candidate.")
    detected_tech_stack: List[str] = Field(..., description="Extracted core competencies.")
    first_question: str = Field(..., description="The first question to ask the candidate.")

class DigestItem(BaseModel):
    topic: str
    status: str = Field(..., description="Competent, Theoretical, Weak, or Unverified")
    summary: str = Field(..., description="A 1-sentence analytical assessment of their response.")

class TechLeadDigest(BaseModel):
    candidate_name: str
    overall_evaluation: str = Field(..., description="A concise executive summary of the candidate's screening performance.")
    strengths: List[str] = Field(..., description="Up to 3 specific areas where the candidate demonstrated solid practical experience.")
    gaps: List[str] = Field(..., description="Any areas where knowledge was theoretical, superficial, or avoided.")
    evaluation_matrix: List[DigestItem] = Field(..., description="The topic-by-topic analysis.")
    suggested_follow_ups: List[str] = Field(..., description="2-3 highly targeted conceptual/practical questions for the Tech Lead round.")

class SessionStatusResponse(BaseModel):
    session_id: str
    current_question_index: int
    total_questions: int
    is_completed: bool
    next_prompt: str  # Could be the next core question, a follow-up, or a completion message
    transcript_so_far: List[Dict[str, str]]
    digest: Optional[TechLeadDigest] = None

# --- Core Helper Functions ---

def transcribe_audio_payload(audio_bytes: bytes, content_type: str) -> str:
    """
    Leverages Gemini 2.5's native multimodal understanding to directly parse 
    the audio bytes into a clean text transcript.
    """
    # Map common web recording MIME types safely
    mime_type = "audio/webm"
    if "wav" in content_type:
        mime_type = "audio/wav"
    elif "mp4" in content_type:
        mime_type = "audio/mp4"

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type,
                ),
                "Transcribe this interview audio snippet exactly. Clean up verbal ticks (like 'um', 'uh', 'like') "
                "to yield a readable transcript, but do not alter the technical terminology or meaning. "
                "Return only the transcription. Do not explain, introduce, or format anything else."
            ]
        )
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal Audio Transcription Failed: {str(e)}")

# --- Endpoints ---

@app.post("/api/v1/interview/initialize", response_model=InterviewInitializationResponse)
async def initialize_interview(
    job_description: str = Form(..., description="The pasted job description."),
    file: UploadFile = File(..., description="The candidate's PDF resume file.")
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="The file submission must be a PDF.")

    try:
        # 1. Parse PDF
        contents = await file.read()
        file_stream = io.BytesIO(contents)
        parsing_result = md_converter.convert_stream(file_stream, file_extension=".pdf")
        resume_markdown = parsing_result.text_content

        # 2. Setup system prompt matching our verified friendly recruiter persona
        system_instruction = (
            "You are a friendly, professional, and experienced HR Recruiter conducting a very first-round, "
            "initial screening interview. Your goal is to assess basic alignment and high-level tech stack familiarity.\n\n"
            "MANDATES:\n"
            "1. Generate exactly 5 conversational, approachable screening questions suitable for a first-round HR call.\n"
            "2. Focus questions around checking actual tools and frameworks claimed on the resume against the job description.\n"
            "3. Do not ask heavy programming or systems optimization scenarios.\n"
            "4. For every question, write structured, highly binary evaluation criteria detailing what a 'Strong Response' vs 'Shallow Response' looks like."
        )

        user_prompt = f"""
        [JOB OPENING POSTING]
        {job_description}

        [CANDIDATE RESUME MARKDOWN]
        {resume_markdown}
        """

        # Define an internal target schema for Gemini specifically for the initialization prompt
        class GeminiInitSchema(BaseModel):
            candidate_name: str
            detected_tech_stack: List[str]
            questions: List[ScreeningQuestion]

        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=GeminiInitSchema,
                temperature=0.3,
            ),
        )

        init_data = GeminiInitSchema.model_validate_json(response.text)

        # 3. Spin up an active session state instance
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "candidate_name": init_data.candidate_name,
            "detected_tech_stack": init_data.detected_tech_stack,
            "questions": init_data.questions,
            "current_question_index": 0,
            "chat_history": [],  # Format: [{"role": "assistant"|"candidate", "text": "..."}]
            "evaluation_history": [], # Format: [{"question_id": X, "transcript": "...", "notes": "..."}]
            "is_completed": False
        }

        # Seed the first question into chat history
        first_q = init_data.questions[0].question_text
        active_sessions[session_id]["chat_history"].append({"role": "assistant", "text": first_q})

        return InterviewInitializationResponse(
            session_id=session_id,
            candidate_name=init_data.candidate_name,
            detected_tech_stack=init_data.detected_tech_stack,
            first_question=first_q
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview Initialization Failure: {str(e)}")
    finally:
        await file.close()


@app.post("/api/v1/interview/{session_id}/respond", response_model=SessionStatusResponse)
async def submit_candidate_response(
    session_id: str,
    file: Optional[UploadFile] = File(None, description="The raw audio recording file."),
    text_response: Optional[str] = Form(None, description="The plaintext typed response from the candidate.")
):
    """
    Receives candidate input via text OR audio file payload, tracks progress,
    and dynamically routes the conversation logic.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Active interview session not found.")

    session = active_sessions[session_id]
    if session["is_completed"]:
         raise HTTPException(status_code=400, detail="This interview has already finished.")

    # 1. Determine Input Channel (Prioritize text fallback for testing)
    transcription = ""
    if text_response and text_response.strip():
        transcription = text_response.strip()
    elif file:
        audio_data = await file.read()
        transcription = transcribe_audio_payload(audio_data, file.content_type)
    else:
        raise HTTPException(
            status_code=400, 
            detail="You must provide either a text_response or an audio file submission."
        )
    
    if not transcription:
        transcription = "[Candidate provided an empty answer]"

    # Save candidate response to chat history
    session["chat_history"].append({"role": "candidate", "text": transcription})

    # 2. Evaluate current question progress
    current_idx = session["current_question_index"]
    current_question = session["questions"][current_idx]

    evaluation_prompt = f"""
    You are the HR Recruiter evaluating a candidate's response to your screening question.
    
    QUESTION ASKED: "{current_question.question_text}"
    EVALUATION CRITERIA:
    {current_question.evaluation_criteria}
    
    CANDIDATE'S RESPONSE: "{transcription}"
    
    Determine if this response is a "PASS" or "SHALLOW".
    
    Respond in strict JSON:
    {{
      "status": "PASS" | "SHALLOW",
      "analysis": "1-sentence reason.",
      "follow_up_prompt": "Friendly follow-up if SHALLOW, otherwise empty."
    }}
    """

    class EvaluatorSchema(BaseModel):
        status: str
        analysis: str
        follow_up_prompt: str

    eval_response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=evaluation_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvaluatorSchema,
            temperature=0.2,
        )
    )
    
    evaluation = EvaluatorSchema.model_validate_json(eval_response.text)

    # 3. Conversation Routing
    next_prompt = ""
    current_chat_turns = [turn for turn in session["chat_history"] if turn["role"] == "candidate"]
    turn_count_for_current_q = len(current_chat_turns) - current_idx
    
    if evaluation.status == "SHALLOW" and turn_count_for_current_q < 2:
        next_prompt = evaluation.follow_up_prompt
        session["chat_history"].append({"role": "assistant", "text": next_prompt})
    else:
        session["evaluation_history"].append({
            "topic": current_question.topic,
            "question_text": current_question.question_text,
            "transcription": transcription,
            "criteria": current_question.evaluation_criteria
        })
        
        session["current_question_index"] += 1
        next_idx = session["current_question_index"]
        
        if next_idx < len(session["questions"]):
            next_prompt = session["questions"][next_idx].question_text
            session["chat_history"].append({"role": "assistant", "text": next_prompt})
        else:
            session["is_completed"] = True
            next_prompt = "Thank you so much! That completes our screening. I am compiling your summary digest for our engineering team now."
            session["chat_history"].append({"role": "assistant", "text": next_prompt})

    digest_output = None
    if session["is_completed"]:
        digest_output = compile_tech_lead_digest(session)
        session["digest"] = digest_output

    return SessionStatusResponse(
        session_id=session_id,
        current_question_index=session["current_question_index"],
        total_questions=len(session["questions"]),
        is_completed=session["is_completed"],
        next_prompt=next_prompt,
        transcript_so_far=session["chat_history"],
        digest=digest_output
    )

def compile_tech_lead_digest(session: Dict[str, Any]) -> TechLeadDigest:
    """
    Passes the complete chat transcripts and performance metrics to Claude/Gemini
    to compile the final target structured portfolio deliverable: The Tech Lead Digest.
    """
    history_blocks = []
    for item in session["evaluation_history"]:
        history_blocks.append(
            f"TOPIC: {item['topic']}\n"
            f"QUESTION: {item['question_text']}\n"
            f"RESPONSE PROVIDED: {item['transcription']}\n"
            f"EVALUATION CRITERIA: {item['criteria']}\n"
            "--------------------"
        )
    formatted_evals = "\n".join(history_blocks)

    digest_prompt = f"""
    You are an elite, highly critical Principal Engineer compiled feedback on a candidate's HR technical screen.
    Review the conversation transcripts and criteria evaluation steps below. Generate a concise, sharp tech lead brief.
    
    CANDIDATE: {session['candidate_name']}
    DETECTOR COMPETENCY MATRIX: {", ".join(session['detected_tech_stack'])}
    
    EVALUATED RESPONSES:
    {formatted_evals}
    
    INSTRUCTIONS FOR COMPILING THE DIGEST:
    1. Overall Evaluation: Write a 2-3 sentence technical assessment of their actual capability. Skip empty fluff. Be direct.
    2. Strengths: Highlight up to 3 core areas where they displayed solid, lived, practical engineering context (using specific details they mentioned).
    3. Gaps: Detail areas where they were superficial, book-smart but lacking hand-on execution, or completely dodged complexity.
    4. Matrix: For each topic, grade their status: "Competent" (knows practical mechanics), "Theoretical" (knows definitions only), "Weak" (incorrect or confused), or "Unverified" (skipped).
    5. Suggested Follow-ups: Write 2-3 targeted technical questions specifically designed for the next stage interviewer (the Tech Lead) to test their suspicious or weak areas.
    """

    response = ai_client.models.generate_content(
        model='gemini-2.5-flash', # Or Claude-3.5-sonnet/Gemini-Pro if you move to production
        contents=digest_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TechLeadDigest,
            temperature=0.3,
        )
    )

    return TechLeadDigest.model_validate_json(response.text)