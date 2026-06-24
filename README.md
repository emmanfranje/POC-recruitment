
# AI-powered HR Initial Screening Terminal (WIP)

Created with initial HR interviews in mind, this is a stateful candidate screening application that parses a resume PDFs and cross-references them against job descriptions and requirements, then compiles a structured summary at the end.

## 💻Current Core Features
### Document Parsing
- Uses Microsoft MarkItDown library to cleanly extract layout heirarchies, tables and lists from a candidate PDF resume into clean markdown.
- The parsed profile is fed into Gemini to generate 5 non-generic, hihg-level HR allignment questions and pairs them with an evaluation criteria.

### Stateful Conversational Interaction
- The candidate responds by using the microphone. The client captures the input as ```.webm``` chunks and streams them to the backend.
- The backend leverages Gemini's multimodal capabilities to transcribe raw audio directly, completely bypassing the need for a separate speech-to-text API.*
- If a candidate provides a shallow or overly brief response, the response is referenced against the evaluation criteria and a follow-up question will be launched to dig deeper before moving to the next question.

### Technical Summary
- At session close, the complete transcript, context data, and grading history are compiled into a technical brief.
- This generates a candidate summary, absolute strengths, key knowledge gaps/risks, a grading matrix (grades topics as Competent, Theoretical, Weak, etc), as well as suggested interview questions tailored for technical interviews should the candidate proceed to the next application process.

## 🏗️Architecture
<img width="674" height="696" alt="image" src="https://github.com/user-attachments/assets/192938e2-fc67-4704-b924-aa61bd50fa0c" />


## 💡Future Planned Features
- **Full audio input:** currently a text input is added in addition to audio input. Streaming audio into Gemini consumes a significant amount of tokens and the text input serves as a temporary solution in the current development phase.
- **Initial HR Interview Emulation:** the vision for the end goal is to emulate having an interview with HR for an initial screening. To do this, a webcam input is also planned. The AI interviewer will also be fitted with text-to-speech API.

## 🛠️Tech Stack

**Backend:** Python 3.10, FastAPI

**Frontend:** Tailwind CSS

**AI:** Gemini 2.5 flash

**Document Parsing:** MarkItDown, Pydantic

## 🚀Quick Start (Local Development)
### Clone the Repository
```bash
https://github.com/emmanfranje/POC-recruitment
cd poc-recruitement
```

### Configure Environment
Create a ```.env``` file in the root directory:
```bash
GEMINI_API_KEY=your_api_key_here
```

### Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Run the FastAPI Server
```bash
uvicorn main:app --reload
```
### Open the Client Terminal
Open ```index.html``` in your browser (ideally using a local server like VS Code's Live Server extension or running ```python3 -m http.server 8080```).
## Engineering Decisions
- To optimize Gemini's rate limits, a text input is implemented alongside the audio input. Audio streams to Gemini consumes significant amounts of tokens and a text input is a temporary solution that fits the current development phase.
- **Microsoft MarkItDown:** Uploading PDFs directly to an LLM is not efficient in both token usage and performance. AI can get confused on the visual layout. LLMs are native in markdown, and MarkItDown preserves the visual layout and structural elements like bullet points, tables, and headers to ensure AI can easily read the document's heirarchy
