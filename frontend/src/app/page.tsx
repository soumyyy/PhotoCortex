"use client";  // This is a client component

import React, { useState, useEffect } from "react";
import "../styles/table.css";
import { Info, Check, AlertCircle } from 'lucide-react';
import Image from 'next/image';
import { Tabs } from '@/components/Tabs';
import { PhotoGrid } from '@/components/PhotoGrid';
import { PeopleGrid } from '@/components/PeopleGrid';
import type { Person, ProcessingResult, StreamResponse } from '@/types';

interface Metadata {
  dimensions: {
    width: number;
    height: number;
  };
  exif?: {
    camera_make?: string;
    camera_model?: string;
    date_taken?: string;
    exposure?: string;
    f_stop?: string;
    iso?: number;
    focal_length?: string;
    gps?: {
      latitude?: number;
      longitude?: number;
      altitude?: number;
    };
  };
  file_size?: number;
  file_type?: string;
  last_modified?: string;
}

interface Face {
  bbox: number[];
  confidence: number;
  face_image: string;
  filename: string;
}

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processedImages, setProcessedImages] = useState<ProcessingResult[]>([]);
  const [uniquePeople, setUniquePeople] = useState<Person[]>([]);

  // Clear state on component mount
  useEffect(() => {
    return () => {
      setProcessedImages([]);
      setUniquePeople([]);
    }
  }, []);

  const handleAnalyzeFolder = async () => {
    setIsLoading(true);
    setError(null);
    setProcessedImages([]); // Clear previous results
    setUniquePeople([]);    // Clear previous results

    try {
      const response = await fetch('http://localhost:8000/analyze-folder');
      const reader = response.body?.getReader();
      
      if (!reader) {
        throw new Error('No reader available');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data: StreamResponse = JSON.parse(line);
              //console.log('Received data:', data); // Debugging

              if (data.type === 'processing' && data.result) {
                // Handle individual image processing results
                setProcessedImages(prev => [...prev, data.result!]);
              } else if (data.type === 'unique_people' && data.people) {
                // Handle clustered results
                setUniquePeople(data.people);
              } else if (data.type === 'error') {
                setError(data.message || 'An unknown error occurred');
              }
            } catch (parseError) {
              console.error('Error parsing JSON:', parseError);
              setError('Error parsing server response');
            }
          }
        }
      }
    } catch (fetchError) {
      console.error('Error fetching data:', fetchError);
      setError('Failed to connect to the server');
    } finally {
      setIsLoading(false);
    }
  };

  // Add effect to monitor state changes
  useEffect(() => {
    console.log('Unique people updated:', uniquePeople.length);
  }, [uniquePeople]);

  useEffect(() => {
    console.log('Processed images updated:', processedImages.length);
  }, [processedImages]);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-4">Photo Analysis</h1>
        <button
          onClick={handleAnalyzeFolder}
          disabled={isLoading}
          className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          {isLoading ? 'Analyzing...' : 'Analyze Photos'}
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500 text-red-500 p-4 rounded mb-8">
          {error}
        </div>
      )}

      {(processedImages.length > 0 || uniquePeople.length > 0) && (
        <Tabs labels={['Photos', 'People']}>
          <PhotoGrid processedImages={processedImages} />
          <PeopleGrid uniquePeople={uniquePeople} />
        </Tabs>
      )}
    </main>
  );
}