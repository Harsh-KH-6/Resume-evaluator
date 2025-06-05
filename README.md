# AI Resume Evaluator

A modern web application that helps job seekers evaluate their resumes against job descriptions using AI-powered analysis. The application provides detailed feedback on resume content, format, and ATS (Applicant Tracking System) compatibility.

## Features

- Resume upload and parsing
- Job description analysis
- ATS scoring system
- Detailed feedback and suggestions
- Score breakdown by categories:
  - Keyword Match
  - Format Score
  - Experience Score
  - Skills Score
  - Achievement Score

## Tech Stack

### Frontend
- React with TypeScript
- Modern UI/UX design
- Responsive layout

### Backend
- Python FastAPI
- PDF parsing and text extraction
- AI-powered analysis
- RESTful API

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn
- Git

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create an uploads directory:
   ```bash
   mkdir uploads
   ```

5. Start the backend server:
   ```bash
   uvicorn app:app --reload
   ```
   The backend will run on http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:
   ```bash
   npm start
   # or
   yarn start
   ```
   The frontend will run on http://localhost:3000

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Upload your resume (PDF format)
3. Enter the job description details:
   - Job Title
   - Job Description
   - Required Skills
   - Years of Experience
4. Click "Evaluate Resume" to get your analysis
5. Review your ATS score and improvement suggestions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Improvements

- [ ] Add support for more resume formats
- [ ] Implement user accounts and resume history
- [ ] Add more detailed feedback categories
- [ ] Integrate with job boards
- [ ] Add resume templates and builder
- [ ] Implement AI-powered resume writing suggestions
- [ ] Add support for multiple languages
- [ ] Implement real-time collaboration features

## Contact

K.H.Harsh (harsh06pb@gmail.com)

