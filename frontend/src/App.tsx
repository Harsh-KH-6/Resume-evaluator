import React, { useState } from 'react';
import UploadForm from './components/UploadForm';

interface JobDescription {
  title: string;
  description: string;
  required_skills: string[];
  experience_years?: number;
}

interface EvaluationResult {
  ats_score: number;
  suggestions: string[];
  score_breakdown: {
    keyword_match: number;
    format_score: number;
    experience_score: number;
    skills_score: number;
    achievement_score: number;
  };
}

function App() {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [resumePath, setResumePath] = useState<string | null>(null);
  const [jobDescription, setJobDescription] = useState<JobDescription>({
    title: '',
    description: '',
    required_skills: [],
    experience_years: undefined
  });
  const [evaluationResult, setEvaluationResult] = useState<EvaluationResult | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    setError(null);
    setUploadSuccess(false);
    setEvaluationResult(null);

    try {
      console.log('Starting file upload...', file.name);
      const formData = new FormData();
      formData.append('file', file);

      console.log('Sending request to backend...');
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);
      const responseData = await response.json();
      console.log('Response data:', responseData);

      if (!response.ok) {
        throw new Error(responseData.detail || `Server error: ${response.status}`);
      }

      console.log('Upload successful:', responseData);
      setUploadSuccess(true);
      setResumePath(responseData.file_path);
    } catch (err) {
      console.error('Upload error:', err);
      if (err instanceof Error) {
        if (err.message === 'Failed to fetch') {
          setError('Unable to connect to the server. Please make sure the backend server is running at http://localhost:8000');
        } else {
          setError(err.message);
        }
      } else {
        setError('An unexpected error occurred during upload');
      }
    } finally {
      setIsUploading(false);
    }
  };

  const handleEvaluate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!resumePath) return;

    setIsEvaluating(true);
    setError(null);

    try {
      // Format the job description data
      const formattedJobDescription = {
        ...jobDescription,
        required_skills: jobDescription.required_skills.filter(Boolean), // Remove any empty strings
        experience_years: jobDescription.experience_years ? Number(jobDescription.experience_years) : undefined
      };

      // Construct the URL with the resume_path as a query parameter
      const evaluateUrl = `http://localhost:8000/evaluate?resume_path=${encodeURIComponent(resumePath)}`;

      const response = await fetch(evaluateUrl, { // Use the new URL
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Send the formattedJobDescription directly as the body
        body: JSON.stringify(formattedJobDescription),
      });

      // Log the response status
      console.log('Response status:', response.status);

      const data = await response.json();
      console.log('Response data:', data);

      if (!response.ok) {
        // Check if the error data has a detail field which is an array
        if (response.status === 422 && Array.isArray(data.detail)) {
          // Format validation errors into a readable string
          const validationErrors = data.detail.map((err: any) => {
            // Adjust loc[0] as it will now refer to body fields directly
            const loc = err.loc.length > 1 ? `${err.loc[0]} -> ${err.loc[1]}` : err.loc[0];
            return `${loc}: ${err.msg}`;
          }).join(', ');
          throw new Error(`Validation failed: ${validationErrors}`);
        } else {
          // Handle other server errors
          throw new Error(data.detail || `Server error: ${response.status}`);
        }
      }

      setEvaluationResult(data);
    } catch (err) {
      console.error('Evaluation error:', err);
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred during evaluation');
      }
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleSkillsChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    // Split by comma, trim each skill, and filter out any resulting empty strings.
    // This handles cases with multiple commas or leading/trailing spaces around skills.
    const skills = e.target.value
      .split(',')
      .map(skill => skill.trim())
      .filter(skill => skill !== ''); // Explicitly filter out empty strings

    console.log('Processed skills:', skills); // Debug log
    setJobDescription(prev => ({ ...prev, required_skills: skills }));
  };

  return (
    <div className="container">
      <header style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', color: '#333', marginBottom: '0.5rem' }}>
          Resume Evaluator
        </h1>
        <p style={{ color: '#666' }}>
          Upload your resume to get instant feedback and scoring
        </p>
      </header>

      <main>
        {/* Upload Form */}
        <div>
          <UploadForm 
            onUpload={handleUpload}
            isUploading={isUploading}
            error={error}
          />
          {uploadSuccess && (
            <div className="success-section">
              <p className="success-message">
                Resume uploaded successfully! Please provide job details for evaluation.
              </p>
              
              {/* Job Description Form */}
              <form onSubmit={handleEvaluate} className="job-description-form">
                <div className="form-group">
                  <label htmlFor="jobTitle">Job Title:</label>
                  <input
                    type="text"
                    id="jobTitle"
                    value={jobDescription.title}
                    onChange={(e) => setJobDescription(prev => ({ ...prev, title: e.target.value.trim() }))}
                    required
                    className="form-input"
                    placeholder="e.g., Software Engineer"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="jobDescription">Job Description:</label>
                  <textarea
                    id="jobDescription"
                    value={jobDescription.description}
                    onChange={(e) => setJobDescription(prev => ({ ...prev, description: e.target.value }))}
                    required
                    className="form-input"
                    rows={6}
                    style={{ whiteSpace: 'pre-wrap' }}
                    placeholder="Enter the job description... (You can use bullet points with • or -)"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="requiredSkills">Required Skills (comma-separated):</label>
                  <textarea
                    id="requiredSkills"
                    value={jobDescription.required_skills.join(', ')}
                    onChange={handleSkillsChange}
                    required
                    className="form-input"
                    rows={2}
                    placeholder="e.g., Python, JavaScript, React"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="experienceYears">Required Years of Experience:</label>
                  <input
                    type="number"
                    id="experienceYears"
                    value={jobDescription.experience_years || ''}
                    onChange={(e) => {
                      const value = e.target.value ? parseInt(e.target.value) : undefined;
                      console.log('Experience years value:', value); // Debug log
                      setJobDescription(prev => ({ ...prev, experience_years: value }));
                    }}
                    min="0"
                    className="form-input"
                    placeholder="e.g., 2"
                  />
                </div>

                <button 
                  type="submit" 
                  className="evaluate-button"
                  disabled={isEvaluating || !jobDescription.title || !jobDescription.description || jobDescription.required_skills.length === 0}
                >
                  {isEvaluating ? 'Evaluating...' : 'Evaluate Resume'}
                </button>
              </form>
            </div>
          )}

          {/* Evaluation Results */}
          {evaluationResult && (
            <div className="evaluation-results">
              <h2>Resume Analysis</h2>
              
              {/* ATS Score */}
              <div className="ats-score-section">
                <h3>ATS Score: {evaluationResult.ats_score}%</h3>
                <div className="score-breakdown">
                  <h4>Score Breakdown:</h4>
                  <ul>
                    <li>Keyword Match: {evaluationResult.score_breakdown.keyword_match}%</li>
                    <li>Format Score: {evaluationResult.score_breakdown.format_score}%</li>
                    <li>Experience Score: {evaluationResult.score_breakdown.experience_score}%</li>
                    <li>Skills Score: {evaluationResult.score_breakdown.skills_score}%</li>
                    <li>Achievement Score: {evaluationResult.score_breakdown.achievement_score}%</li>
                  </ul>
                </div>
              </div>

              {/* Suggestions */}
              <div className="suggestions-section">
                <h3>Resume Improvements Needed:</h3>
                <ul>
                  {evaluationResult.suggestions.map((suggestion, index) => (
                    <li key={index}>{suggestion}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer style={{ 
        marginTop: '3rem', 
        textAlign: 'center', 
        color: '#666',
        fontSize: '0.875rem'
      }}>
        <p>© 2024 Resume Evaluator. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
