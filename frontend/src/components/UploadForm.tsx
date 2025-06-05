import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadFormProps {
  onUpload: (file: File) => Promise<void>;
  isUploading: boolean;
  error: string | null;
}

const UploadForm: React.FC<UploadFormProps> = ({ onUpload, isUploading, error }) => {
  const [isDragging, setIsDragging] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    multiple: false,
    onDragEnter: () => setIsDragging(true),
    onDragLeave: () => setIsDragging(false)
  });

  return (
    <div className="upload-form">
      <h2>Upload Your Resume</h2>
      <div
        {...getRootProps()}
        className={`upload-area ${isDragActive ? 'dragging' : ''}`}
      >
        <input {...getInputProps()} />
        {isUploading ? (
          <div>
            <span className="loading-spinner"></span>
            Uploading...
          </div>
        ) : isDragActive ? (
          <p>Drop your resume here...</p>
        ) : (
          <p>Drag and drop your resume here, or click to select a file</p>
        )}
      </div>
      {error && <p className="error-message">{error}</p>}
      <p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#666' }}>
        Supported formats: PDF, DOCX
      </p>
    </div>
  );
};

export default UploadForm; 