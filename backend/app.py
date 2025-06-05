from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from typing import Optional, List
from pydantic import BaseModel
import shutil
from pathlib import Path
from extract import ResumeExtractor
from score import ResumeScorer
import re
from collections import Counter

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Resume Evaluator API",
    description="API for evaluating resumes and providing feedback",
    version="1.0.0"
)

# Configure CORS with more permissive settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize the resume scorer
resume_scorer = ResumeScorer()

class JobDescription(BaseModel):
    title: str
    description: str
    required_skills: List[str]
    experience_years: Optional[int] = None
    keywords: Optional[List[str]] = None  # Additional keywords to look for
    industry: Optional[str] = None  # Industry-specific requirements

class ResumeAnalysis:
    def __init__(self):
        self.common_ats_keywords = {
            'technical': ['python', 'java', 'javascript', 'sql', 'aws', 'azure', 'docker', 'kubernetes', 'git', 'agile', 'scrum'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem-solving', 'time management', 'adaptability'],
            'education': ['bachelor', 'master', 'phd', 'degree', 'certification', 'diploma'],
            'experience': ['experience', 'years', 'worked', 'managed', 'led', 'developed', 'implemented'],
            'achievements': ['achieved', 'increased', 'decreased', 'improved', 'developed', 'created', 'implemented']
        }

    def analyze_resume(self, text: str, job_description: JobDescription) -> dict:
        # Clean and normalize text
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        word_freq = Counter(words)

        # Calculate ATS score components
        scores = {
            'keyword_match': self._calculate_keyword_match(text, job_description),
            'format_score': self._check_format(text),
            'experience_score': self._check_experience(text, job_description),
            'skills_score': self._check_skills(text, job_description),
            'achievement_score': self._check_achievements(text)
        }

        # Calculate overall ATS score (weighted average)
        weights = {
            'keyword_match': 0.3,
            'format_score': 0.2,
            'experience_score': 0.2,
            'skills_score': 0.2,
            'achievement_score': 0.1
        }
        
        overall_score = sum(score * weights[component] for component, score in scores.items())

        # Generate suggestions
        suggestions = self._generate_suggestions(text, job_description, scores)

        return {
            'ats_score': round(overall_score * 100, 1),  # Convert to percentage
            'score_breakdown': {
                component: round(score * 100, 1) for component, score in scores.items()
            },
            'suggestions': suggestions,
            'keyword_analysis': {
                'matched_keywords': self._find_matched_keywords(text, job_description),
                'missing_keywords': self._find_missing_keywords(text, job_description)
            }
        }

    def _calculate_keyword_match(self, text: str, job_description: JobDescription) -> float:
        # Combine all keywords to look for
        keywords = set(job_description.required_skills)
        if job_description.keywords:
            keywords.update(job_description.keywords)
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword.lower() in text)
        return min(matches / len(keywords), 1.0) if keywords else 0.0

    def _check_format(self, text: str) -> float:
        score = 1.0
        
        # Check for common formatting issues
        if len(text.split('\n')) < 10:  # Too short
            score *= 0.8
        if len(text) > 2000:  # Too long
            score *= 0.9
        if not re.search(r'\b(experience|education|skills)\b', text.lower()):
            score *= 0.7  # Missing important sections
            
        return score

    def _check_experience(self, text: str, job_description: JobDescription) -> float:
        if not job_description.experience_years:
            return 1.0

        # Look for experience indicators
        experience_patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of)?\s*experience',
            r'experience:\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:in)?\s*the\s*field'
        ]
        
        for pattern in experience_patterns:
            match = re.search(pattern, text.lower())
            if match:
                years = int(match.group(1))
                return min(years / job_description.experience_years, 1.0)
        
        return 0.5  # Default score if no experience found

    def _check_skills(self, text: str, job_description: JobDescription) -> float:
        if not job_description.required_skills:
            return 1.0

        # Count how many required skills are mentioned
        skills_found = sum(1 for skill in job_description.required_skills 
                         if skill.lower() in text.lower())
        return skills_found / len(job_description.required_skills)

    def _check_achievements(self, text: str) -> float:
        # Look for achievement indicators
        achievement_indicators = [
            r'increased',
            r'decreased',
            r'improved',
            r'achieved',
            r'developed',
            r'created',
            r'implemented',
            r'reduced',
            r'optimized',
            r'enhanced'
        ]
        
        matches = sum(1 for indicator in achievement_indicators 
                     if re.search(r'\b' + indicator + r'\b', text.lower()))
        return min(matches / 5, 1.0)  # Cap at 1.0 for 5 or more achievements

    def _generate_suggestions(self, text: str, job_description: JobDescription, scores: dict) -> List[str]:
        suggestions = []
        
        # Keyword match suggestions
        if scores['keyword_match'] < 0.7:
            missing_keywords = self._find_missing_keywords(text, job_description)
            if missing_keywords:
                suggestions.append(f"Add these keywords to your resume: {', '.join(missing_keywords[:3])}")
        
        # Format suggestions
        if scores['format_score'] < 0.8:
            if len(text.split('\n')) < 10:
                suggestions.append("Your resume seems too short. Consider adding more details about your experience and achievements.")
            if len(text) > 2000:
                suggestions.append("Your resume might be too long. Consider being more concise and focusing on relevant achievements.")
            if not re.search(r'\b(experience|education|skills)\b', text.lower()):
                suggestions.append("Make sure to include clear sections for Experience, Education, and Skills.")
        
        # Experience suggestions
        if scores['experience_score'] < 0.8 and job_description.experience_years:
            suggestions.append(f"Highlight your {job_description.experience_years}+ years of experience more prominently.")
        
        # Skills suggestions
        if scores['skills_score'] < 0.8:
            missing_skills = [skill for skill in job_description.required_skills 
                            if skill.lower() not in text.lower()]
            if missing_skills:
                suggestions.append(f"Add these required skills to your resume: {', '.join(missing_skills[:3])}")
        
        # Achievement suggestions
        if scores['achievement_score'] < 0.6:
            suggestions.append("Add more quantifiable achievements to your resume. Use action verbs and include specific results.")
        
        return suggestions

    def _find_matched_keywords(self, text: str, job_description: JobDescription) -> List[str]:
        keywords = set(job_description.required_skills)
        if job_description.keywords:
            keywords.update(job_description.keywords)
        return [keyword for keyword in keywords if keyword.lower() in text.lower()]

    def _find_missing_keywords(self, text: str, job_description: JobDescription) -> List[str]:
        keywords = set(job_description.required_skills)
        if job_description.keywords:
            keywords.update(job_description.keywords)
        return [keyword for keyword in keywords if keyword.lower() not in text.lower()]

# Initialize resume analyzer
resume_analyzer = ResumeAnalysis()

@app.get("/")
async def root():
    return {"message": "Welcome to Resume Evaluator API"}

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a resume file (PDF or DOCX)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx']:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF and DOCX files are allowed."
        )
    
    try:
        # Save the file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract text from the resume
        extractor = ResumeExtractor()
        text = extractor.extract_text(file_path)
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the resume. The file might be corrupted or empty."
            )
        
        # Clean the extracted text
        cleaned_text = extractor.clean_text(text)
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "file_path": str(file_path),
            "extracted_text": cleaned_text
        }
    except Exception as e:
        # Clean up the file if something goes wrong
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate_resume(
    job_description: JobDescription,
    resume_path: str
):
    """
    Evaluate a resume against a job description
    """
    try:
        # Extract text from resume
        extractor = ResumeExtractor()
        resume_text = extractor.extract_text(resume_path)
        if not resume_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the resume"
            )
        
        # Clean the text
        cleaned_text = extractor.clean_text(resume_text)
        
        # Get AI analysis
        analysis = resume_analyzer.analyze_resume(cleaned_text, job_description)
        
        # Get basic scoring
        result = resume_scorer.score_resume(
            resume_text=cleaned_text,
            job_description=job_description.description,
            required_skills=job_description.required_skills,
            required_years=job_description.experience_years
        )
        
        return {
            "message": "Evaluation completed successfully",
            "resume_path": resume_path,
            "job_title": job_description.title,
            "ats_score": analysis['ats_score'],
            "score_breakdown": analysis['score_breakdown'],
            "suggestions": analysis['suggestions'],
            "keyword_analysis": analysis['keyword_analysis'],
            **result
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Resume file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 