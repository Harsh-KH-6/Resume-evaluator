import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_upload_resume(file_path: str) -> dict:
    """
    Test the resume upload endpoint
    """
    url = f"{BASE_URL}/upload"
    
    with open(file_path, "rb") as f:
        files = {"file": (Path(file_path).name, f, "application/pdf")}
        response = requests.post(url, files=files)
    
    print("\nUpload Response:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_evaluate_resume(resume_path: str, job_description: dict) -> dict:
    """
    Test the resume evaluation endpoint
    """
    url = f"{BASE_URL}/evaluate"
    
    data = {
        "job_description": job_description,
        "resume_path": resume_path
    }
    
    response = requests.post(url, json=data)
    
    print("\nEvaluation Response:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

if __name__ == "__main__":
    # Test job description
    job_description = {
        "title": "Senior Software Engineer",
        "description": """
        We are looking for a Senior Software Engineer with strong experience in Python and web development.
        The ideal candidate should have experience with modern web frameworks, cloud technologies, and
        database systems. Experience with machine learning and AI is a plus.
        """,
        "required_skills": [
            "python",
            "web development",
            "cloud",
            "database",
            "machine learning"
        ],
        "experience_years": 5
    }
    
    # Test with a sample resume
    # Replace with your test resume path
    test_resume_path = "C:/Users/harsh/OneDrive/Desktop/Resume_Checker/backend/test_resume.pdf"
    
    try:
        # Test upload
        upload_result = test_upload_resume(test_resume_path)
        
        # Test evaluation
        if "file_path" in upload_result:
            evaluation_result = test_evaluate_resume(
                upload_result["file_path"],
                job_description
            )
    except Exception as e:
        print(f"Error during testing: {e}")