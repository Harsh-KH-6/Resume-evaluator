from pathlib import Path
from typing import Optional
import docx
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

class ResumeExtractor:
    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> Optional[str]:
        """
        Extract text from a PDF file
        """
        try:
            text = extract_text(str(file_path))
            return text.strip() if text else None
        except PDFSyntaxError:
            return None
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None

    @staticmethod
    def extract_text_from_docx(file_path: Path) -> Optional[str]:
        """
        Extract text from a DOCX file
        """
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip() if text else None
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return None

    @classmethod
    def extract_text(cls, file_path: str | Path) -> Optional[str]:
        """
        Extract text from a resume file (PDF or DOCX)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return cls.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and normalizing line endings
        """
        if not text:
            return ""
        
        # Replace multiple newlines with a single newline
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        
        # Replace multiple spaces with a single space
        text = " ".join(text.split())
        
        return text 