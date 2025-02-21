export interface Face {
  filename: string;
  face_image: string;
  confidence: number;
  box: number[];
}

export interface Person {
  person_id: number;
  instances_count: number;
  confidence_score: number;
  representative_face: Face;
  appearances: Face[];
}

export interface ImageMetadata {
  basic: {
    filename: string;
    file_size: string;
    dimensions: string;
    format: string;
    mode: string;
    created: string;
    modified: string;
  };
  exif: {
    [key: string]: string;
  };
  gps: {
    latitude?: string;
    longitude?: string;
    altitude?: string;
  };
}

export interface ProcessingResult {
  filename: string;
  faces: Face[];
  full_image: string;
  metadata: ImageMetadata;
  scene_classification?: {
    scene_type: string;
  };
  object_detections?: string[];
  info?: string;
}

export interface StreamResponse {
  type: 'processing' | 'unique_people' | 'error';
  result?: ProcessingResult;
  people?: Person[];
  message?: string;
} 