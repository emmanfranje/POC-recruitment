import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from markitdown import MarkItDown

# Initialize the FastAPI application
app = FastAPI(
    title="Automated AI Interviewer Core API",
    description="Phase 1: Dynamic Resume Parser and Markdown Converter",
    version="1.0.0"
)

# Enable CORS so your future Next.js frontend can talk to this backend smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the MarkItDown converter engine
md_converter = MarkItDown()

@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Interviewer Backend Core is running."}

@app.post("/api/v1/resume/parse")
async def parse_candidate_resume(file: UploadFile = File(...)):
    """
    Accepts a PDF resume upload, extracts text content natively,
    and returns it formatted as structural Markdown.
    """
    # 1. Enforce strict content type validation
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. This endpoint exclusively accepts PDF files."
        )
    
    try:
        # 2. Read the file file stream asynchronously into memory
        contents = await file.read()
        file_stream = io.BytesIO(contents)
        
        # 3. Stream the bytes through MarkItDown
        # We supply the file_extension hint so MarkItDown activates its PDF parsing route
        result = md_converter.convert_stream(file_stream, file_extension=".pdf")
        
        # 4. Extract the clean structural Markdown string
        markdown_text = result.text_content
        
        return {
            "success": True,
            "filename": file.filename,
            "extracted_content": markdown_text
        }
        
    except Exception as e:
        # Log error or catch parsing failures safely
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing the document structural layers: {str(e)}"
        )
    finally:
        # Ensure the file resource descriptor is tightly released
        await file.close()