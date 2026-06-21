import io
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from markitdown import MarkItDown
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Interview POC",
    description="Phase 1: Resume Parsing & Dynamic Question Generation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

md_converter = MarkItDown()

# Initialize the Gemini Client
# It automatically picks up the GEMINI_API_KEY from our environment variables
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is missing from your environment setup.")
ai_client = genai.Client()

# --- Pydantic Schemas for Structured AI Outputs ---

class ScreeningQuestion(BaseModel):
    id: int = Field(..., description="Unique sequential identifier for the question (1-5).")
    topic: str = Field(..., description="The technical core stack component or requirement being assessed.")
    question_text: str = Field(..., description="The conceptual, scenario-based interview question tailored for the candidate.")
    evaluation_criteria: str = Field(..., description="Detailed instructions for the AI on what a strong, deep answer looks like vs a shallow answer.")

class InterviewInitializationResponse(BaseModel):
    candidate_name: str = Field(..., description="Extracted name of the candidate from the resume.")
    detected_tech_stack: List[str] = Field(..., description="Core frameworks, tools, and languages found in the resume.")
    generated_questions: List[ScreeningQuestion] = Field(..., description="A collection of exactly 5 tailored screening questions.")

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "online"}

@app.post("/api/v1/interview/initialize", response_model=InterviewInitializationResponse)
async def initialize_interview(
    job_description: str = Form(..., description="The complete text string of the job opening posting."),
    file: UploadFile = File(..., description="The candidate's PDF resume file.")
):
    """
    Parses an incoming resume, cross-references it with the target job description,
    and returns 5 structured technical screening questions via Gemini.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="The file submission must be a PDF.")

    try:
        # 1. Convert PDF resume into structural Markdown using MarkItDown
        contents = await file.read()
        file_stream = io.BytesIO(contents)
        parsing_result = md_converter.convert_stream(file_stream, file_extension=".pdf")
        resume_markdown = parsing_result.text_content

        # 2. Construct the system instruction context for the AI
        system_instruction = (
            "You are a friendly, professional, and experienced HR Recruiter conducting a very first-round, "
            "initial screening interview (a 'getting-to-know-you' chat). Your goal is to assess basic alignment, "
            "communication skills, and high-level tech stack familiarity—NOT to deeply grill them on system architecture.\n\n"
            
            "CRITICAL QUESTION GENERATION MANDATES:\n"
            "1. Generate exactly 5 conversational, approachable screening questions suitable for a first-round HR call.\n"
            "2. DO NOT ask heavy technical scenarios, coding questions, or complex optimization problems.\n"
            "3. Keep the focus on introductory topics, such as:\n"
            "   - A high-level overview of their experience with a key tool mentioned in both the job description and resume.\n"
            "   - Their past team dynamics or a time they had to adapt to a new workflow.\n"
            "   - What drew them to this specific role or tech stack.\n"
            "4. Frame the questions naturally, as an HR human would speak (e.g., 'Looking over your profile, I see you've spent some time with React. Could you tell me about a recent project where it was a core part of your stack?').\n\n"
            
            "EVALUATION CRITERIA MANDATES:\n"
            "Keep the criteria tailored for HR metrics:\n"
            "- Strong Response: Confident communication, clear explanation of their high-level role, enthusiasm for the stack, clear timeline alignment.\n"
            "- Shallow Response: Extreme briefness, inability to explain what their previous project actually did, or poor communication clarity."
        )

        # 3. Design the user prompt payload
        user_prompt = f"""
        [JOB DESCRIPTION]
        {job_description}

        [CANDIDATE RESUME MARKDOWN]
        {resume_markdown}
        """

        # 4. Trigger the Gemini API requesting a strict Pydantic JSON structure
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash', # fast, efficient, and great at structured tracking
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=InterviewInitializationResponse,
                temperature=0.2, # Low temperature forces more deterministic, analytical outputs
            ),
        )

        # 5. Parse the validated structured string back directly into our endpoint's schema
        # FastAPI handles transforming this into a perfect JSON output for the client
        return InterviewInitializationResponse.model_validate_json(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Initialization Failure: {str(e)}")
    finally:
        await file.close()