from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class ResumeScorer:
    def __init__(self):
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            # If model not found, download it
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Increased to capture more context
            max_features=10000
        )
        
        # Expanded skill categories with more modern technologies
        self.skill_categories = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
                'rust', 'go', 'scala', 'r', 'matlab', 'perl', 'haskell', 'clojure'
            ],
            'web': [
                'html', 'css', 'sass', 'less', 'react', 'angular', 'vue', 'next.js', 'nuxt.js', 'node.js',
                'django', 'flask', 'express', 'spring', 'asp.net', 'laravel', 'graphql', 'rest api'
            ],
            'database': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'cassandra',
                'dynamodb', 'neo4j', 'elasticsearch', 'firebase', 'cosmos db'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'terraform', 'ansible',
                'jenkins', 'gitlab ci', 'github actions', 'serverless', 'lambda', 'ec2', 's3'
            ],
            'ai_ml': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
                'keras', 'numpy', 'pandas', 'opencv', 'nltk', 'spacy', 'bert', 'gpt', 'transformers',
                'computer vision', 'nlp', 'reinforcement learning'
            ],
            'data_science': [
                'data analysis', 'data visualization', 'power bi', 'tableau', 'excel',
                'statistics', 'probability', 'regression', 'classification', 'clustering',
                'time series', 'forecasting', 'etl', 'data warehousing'
            ],
            'mobile': [
                'android', 'ios', 'react native', 'flutter', 'xamarin', 'swift', 'kotlin',
                'mobile app development', 'app store', 'play store'
            ],
            'devops': [
                'ci/cd', 'git', 'github', 'gitlab', 'bitbucket', 'jenkins', 'travis ci',
                'circle ci', 'monitoring', 'logging', 'prometheus', 'grafana', 'elk stack'
            ],
            'security': [
                'cybersecurity', 'penetration testing', 'security analysis', 'vulnerability assessment',
                'network security', 'application security', 'security compliance', 'owasp'
            ]
        }
        
        # Achievement patterns and metrics
        self.achievement_patterns = {
            'metrics': [
                r'\d+%', r'\$\d+', r'\d+x', r'\d+\+', r'\d+k', r'\d+M',
                r'increased by \d+%', r'decreased by \d+%',
                r'reduced by \d+%', r'improved by \d+%',
                r'saved \$\d+', r'generated \$\d+',
                r'managed \d+', r'led \d+', r'trained \d+'
            ],
            'action_verbs': [
                'achieved', 'increased', 'decreased', 'improved', 'developed',
                'created', 'implemented', 'reduced', 'optimized', 'enhanced',
                'streamlined', 'automated', 'innovated', 'transformed', 'revolutionized',
                'pioneered', 'spearheaded', 'orchestrated', 'architected', 'engineered'
            ]
        }

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills using advanced NLP techniques
        """
        doc = self.nlp(text.lower())
        found_skills = set()
        
        # Extract skills using multiple methods
        # 1. Direct matching with skill categories
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if skill in text.lower():
                    found_skills.add(skill)
        
        # 2. Named Entity Recognition for technical terms
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG', 'WORK_OF_ART']:
                found_skills.add(ent.text.lower())
        
        # 3. Pattern matching for technical phrases
        technical_patterns = [
            r'\b(?:experienced in|proficient in|skilled in|expert in)\s+([^.,]+)',
            r'\b(?:using|with|via)\s+([^.,]+)',
            r'\b(?:developed|built|created|implemented)\s+(?:using|with)\s+([^.,]+)'
        ]
        
        for pattern in technical_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                skill = match.group(1).strip()
                found_skills.add(skill)
        
        return list(found_skills)

    def analyze_achievements(self, text: str) -> Dict[str, Any]:
        """
        Analyze achievements using NLP and pattern matching
        """
        sentences = sent_tokenize(text)
        achievements = []
        metrics_found = []
        action_verbs = []
        
        for sentence in sentences:
            # Check for metrics
            for pattern in self.achievement_patterns['metrics']:
                if re.search(pattern, sentence.lower()):
                    metrics_found.append(sentence)
                    break
            
            # Check for action verbs
            doc = self.nlp(sentence.lower())
            for verb in self.achievement_patterns['action_verbs']:
                if verb in sentence.lower():
                    action_verbs.append(verb)
                    if sentence not in achievements:
                        achievements.append(sentence)
                    break
        
        return {
            'achievements': achievements,
            'metrics_found': metrics_found,
            'action_verbs': list(set(action_verbs)),
            'achievement_score': min(len(achievements) / 5, 1.0)  # Cap at 1.0 for 5+ achievements
        }

    def analyze_experience(self, text: str, required_years: int) -> Dict[str, Any]:
        """
        Advanced experience analysis using NLP
        """
        doc = self.nlp(text)
        experience_info = {
            'years': 0,
            'roles': [],
            'companies': [],
            'responsibilities': []
        }
        
        # Extract years of experience
        experience_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience',
            r'experience\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:in)?\s*the\s*field',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*professional\s*experience'
        ]
        
        for pattern in experience_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                years = int(match.group(1))
                experience_info['years'] = max(experience_info['years'], years)
        
        # Extract roles and companies using NER
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                experience_info['companies'].append(ent.text)
            elif ent.label_ == 'WORK_OF_ART':
                experience_info['roles'].append(ent.text)
        
        # Calculate experience score
        if required_years:
            experience_score = min(experience_info['years'] / required_years, 1.0)
        else:
            experience_score = 1.0
        
        return {
            'experience_info': experience_info,
            'experience_score': experience_score
        }

    def calculate_content_similarity(self, resume_text: str, job_description: str) -> Dict[str, float]:
        """
        Enhanced content similarity analysis using multiple metrics
        """
        try:
            # TF-IDF similarity
            tfidf_matrix = self.vectorizer.fit_transform([resume_text, job_description])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Semantic similarity using spaCy
            doc1 = self.nlp(resume_text)
            doc2 = self.nlp(job_description)
            semantic_similarity = doc1.similarity(doc2)
            
            # Keyword overlap
            resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
            job_words = set(re.findall(r'\b\w+\b', job_description.lower()))
            keyword_overlap = len(resume_words.intersection(job_words)) / len(job_words)
            
            return {
                'tfidf_similarity': float(tfidf_similarity),
                'semantic_similarity': float(semantic_similarity),
                'keyword_overlap': float(keyword_overlap),
                'overall_similarity': float((tfidf_similarity + semantic_similarity + keyword_overlap) / 3)
            }
        except Exception as e:
            print(f"Error calculating content similarity: {e}")
            return {
                'tfidf_similarity': 0.0,
                'semantic_similarity': 0.0,
                'keyword_overlap': 0.0,
                'overall_similarity': 0.0
            }

    def calculate_skill_match(self, resume_text: str, required_skills: List[str]) -> float:
        """
        Calculate skill match score
        """
        resume_skills = self.extract_skills(resume_text)
        resume_skill_set = set(s.lower() for s in resume_skills)
        required_skill_set = set(s.lower() for s in required_skills)
        overlap = resume_skill_set.intersection(required_skill_set)
        if not required_skill_set:
            return 1.0
        return (len(overlap) / len(required_skill_set))

    def generate_feedback(self, resume_text: str, job_description: str, 
                         required_skills: List[str], resume_skills: List[str]) -> Dict[str, Any]:
        """
        Generate detailed feedback using advanced analysis
        """
        # Analyze achievements
        achievement_analysis = self.analyze_achievements(resume_text)
        
        # Analyze experience
        experience_analysis = self.analyze_experience(resume_text, 0)  # Years will be updated later
        
        # Calculate similarities
        similarity_analysis = self.calculate_content_similarity(resume_text, job_description)
        
        feedback = {
            "strengths": [],
            "improvements": [],
            "missing_skills": [],
            "achievement_analysis": achievement_analysis,
            "experience_analysis": experience_analysis,
            "similarity_analysis": similarity_analysis
        }

        # Skill analysis
        required_skill_set = set(skill.lower() for skill in required_skills)
        resume_skill_set = set(skill.lower() for skill in resume_skills)
        missing_skills = required_skill_set - resume_skill_set
        matching_skills = resume_skill_set.intersection(required_skill_set)
        
        if missing_skills:
            feedback["missing_skills"] = list(missing_skills)
            feedback["improvements"].append(
                f"Add these key skills to your resume: {', '.join(list(missing_skills)[:3])}"
            )
        
        if matching_skills:
            feedback["strengths"].append(
                f"Strong technical skills match: {', '.join(matching_skills)}"
            )

        # Achievement analysis
        if achievement_analysis['achievement_score'] < 0.6:
            feedback["improvements"].append(
                "Add more quantifiable achievements. Include specific metrics and results."
            )
        else:
            feedback["strengths"].append(
                f"Good use of achievements and metrics: {len(achievement_analysis['achievements'])} strong examples found"
            )

        # Experience analysis
        if experience_analysis['experience_info']['years'] > 0:
            feedback["strengths"].append(
                f"Clear experience progression: {experience_analysis['experience_info']['years']} years of experience"
            )

        # Content similarity analysis
        if similarity_analysis['overall_similarity'] < 0.5:
            feedback["improvements"].append(
                "Resume content could better align with job description. Consider incorporating more relevant keywords and responsibilities."
            )
        else:
            feedback["strengths"].append(
                "Good alignment with job description requirements"
            )

        return feedback

    def score_resume(self, resume_text: str, job_description: str, 
                    required_skills: List[str], required_years: int = None) -> Dict[str, Any]:
        """
        Comprehensive resume scoring with detailed analysis
        """
        # Extract skills
        resume_skills = self.extract_skills(resume_text)
        
        # Calculate various scores
        skill_match_score = self.calculate_skill_match(resume_text, required_skills)
        experience_analysis = self.analyze_experience(resume_text, required_years or 0)
        similarity_analysis = self.calculate_content_similarity(resume_text, job_description)
        achievement_analysis = self.analyze_achievements(resume_text)
        
        # Calculate overall score with weighted components
        weights = {
            'skill_match': 0.3,
            'experience': 0.2,
            'content_similarity': 0.3,
            'achievements': 0.2
        }
        
        overall_score = (
            skill_match_score * weights['skill_match'] +
            experience_analysis['experience_score'] * weights['experience'] +
            similarity_analysis['overall_similarity'] * weights['content_similarity'] +
            achievement_analysis['achievement_score'] * weights['achievements']
        )
        
        # Generate detailed feedback
        feedback = self.generate_feedback(
            resume_text, job_description, required_skills, resume_skills
        )
        
        return {
            "score": round(overall_score * 100, 1),  # Convert to percentage
            "detailed_scores": {
                "skill_match": round(skill_match_score * 100, 1),
                "experience": round(experience_analysis['experience_score'] * 100, 1),
                "content_similarity": round(similarity_analysis['overall_similarity'] * 100, 1),
                "achievements": round(achievement_analysis['achievement_score'] * 100, 1)
            },
            "feedback": feedback,
            "found_skills": resume_skills,
            "analysis": {
                "achievements": achievement_analysis,
                "experience": experience_analysis,
                "similarity": similarity_analysis
            }
        } 