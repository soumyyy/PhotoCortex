import { useState } from 'react';
import axios from 'axios';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://0.0.0.1:8000/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setAnalysisResult(response.data);
    } catch (error) {
      console.error("Error analyzing image:", error);
    }
  };

  return (
    <div style={{ padding: '2rem' }}>
      <h1>Cortex: Photo Analysis</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleAnalyze}>Analyze Image</button>

      {analysisResult && (
        <div style={{ marginTop: '1rem' }}>
          <h2>Analysis Result:</h2>
          <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}