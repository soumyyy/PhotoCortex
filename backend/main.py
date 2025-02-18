import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageOps, ExifTags
from fastapi.responses import StreamingResponse
import json
import asyncio
import logging
import sys
import exifread
from datetime import datetime
import piexif
from fractions import Fraction
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from collections import defaultdict
import scipy.spatial as spatial
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import face_recognition
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from sklearn.preprocessing import normalize
import gdown  # Add this to requirements.txt
import hashlib
import face_alignment
from face_alignment import LandmarksType
import math
from scipy.spatial.distance import cdist

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MTCNN with optimized parameters
try:
    device = 'cpu'
    logger.info(f"Initializing MTCNN on {device}")
    
    detector = MTCNN(
        keep_all=True,
        device=device,
        min_face_size=30,
        thresholds=[0.6, 0.7, 0.7],  # Slightly relaxed thresholds
        factor=0.707,
        post_process=False,
        select_largest=False
    )
    
    # Test the detector
    test_img = Image.new('RGB', (100, 100))
    test_result = detector.detect(test_img)
    logger.info("MTCNN initialization successful")
    
except Exception as e:
    logger.error(f"Failed to initialize MTCNN: {str(e)}")
    logger.error(f"Python version: {sys.version}")
    logger.error(f"Torch version: {torch.__version__}")
    raise RuntimeError(f"MTCNN initialization failed: {str(e)}")

# Initialize face embedding model (only once)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
if torch.cuda.is_available():
    resnet = resnet.cuda()

@dataclass
class FaceSignature:
    embedding: np.ndarray      # Combined embedding
    filename: str
    face_image: str
    confidence: float
    box: List[float]

    def __post_init__(self):
        """Normalize and validate all features"""
        self.confidence = float(self.confidence)
        self.box = [float(x) for x in self.box]
        self.embedding = normalize(np.asarray(self.embedding).reshape(1, -1))[0]

    def __eq__(self, other) -> bool:
        if not isinstance(other, FaceSignature):
            return False
        return (self.filename == other.filename and 
                np.array_equal(self.box, other.box))

    def __hash__(self) -> int:
        return hash((self.filename, tuple(self.box)))

class RobustFaceClusterer:
    def __init__(self):
        self.face_signatures: List[FaceSignature] = []
        self.min_confidence = 0.92
        self.cache_file = "face_embeddings_cache.json"
        self.cache_lock = threading.Lock()
        self.batch_size = 32
        self.padding_factor = 0.15
        self.verification_threshold = 0.65
        
        # Initialize models
        logger.info("Initializing face recognition models...")
        self.model = DeepFace.build_model("Facenet")
        
        # Simpler face alignment initialization
        self.fa = face_alignment.FaceAlignment(
            2,  # Using integer value instead of enum
            flip_input=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Models initialized successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        self._load_cache()

    def _load_cache(self):
        """Load cached embeddings"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            else:
                self.embedding_cache = {}
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self.embedding_cache = {}

    def _save_cache(self):
        """Save embeddings to cache"""
        try:
            with self.cache_lock:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.embedding_cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def _get_image_hash(self, image_path: str) -> str:
        """Get MD5 hash of image file"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing image {image_path}: {str(e)}")
            return ""

    def _align_face(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align face using modern face alignment"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                if isinstance(image, np.ndarray):
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = image

            # Get landmarks
            landmarks = self.fa.get_landmarks(image_rgb)
            
            if landmarks is None or len(landmarks) == 0:
                logger.warning("No landmarks detected")
                return image, None

            landmarks = landmarks[0]  # Get first face landmarks

            # Get eye coordinates
            left_eye = landmarks[36:42].mean(axis=0)
            right_eye = landmarks[42:48].mean(axis=0)
            
            # Calculate angle for alignment
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = math.degrees(math.atan2(dy, dx))
            
            # Get center of face
            center = ((left_eye + right_eye) / 2).astype(int)
            
            # Create rotation matrix
            M = cv2.getRotationMatrix2D(tuple(center), angle, 1)
            
            # Perform rotation
            aligned = cv2.warpAffine(
                image, 
                M, 
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Rotate landmarks too
            ones = np.ones(shape=(len(landmarks), 1))
            points_ones = np.hstack([landmarks, ones])
            transformed_landmarks = M.dot(points_ones.T).T
            
            return aligned, transformed_landmarks
            
        except Exception as e:
            logger.error(f"Error in face alignment: {str(e)}")
            return image, None

    def _get_face_region(self, image: np.ndarray, box: List[float]) -> np.ndarray:
        """Extract and align face region"""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = map(int, box)
            
            # Calculate padding
            width = x2 - x1
            height = y2 - y1
            padding_w = int(width * self.padding_factor)
            padding_h = int(height * self.padding_factor)
            
            # Apply padding with bounds checking
            x1 = max(0, x1 - padding_w)
            y1 = max(0, y1 - padding_h)
            x2 = min(w, x2 + padding_w)
            y2 = min(h, y2 + padding_h)
            
            # Extract face region
            face_region = image[y1:y2, x1:x2]
            
            # Align face
            aligned_face, landmarks = self._align_face(face_region)
            
            if landmarks is not None:
                # Get face bounding box from landmarks
                min_x, min_y = np.min(landmarks, axis=0)
                max_x, max_y = np.max(landmarks, axis=0)
                
                # Add small margin
                margin = 0.1
                bbox_w = max_x - min_x
                bbox_h = max_y - min_y
                min_x = max(0, min_x - bbox_w * margin)
                min_y = max(0, min_y - bbox_h * margin)
                max_x = min(aligned_face.shape[1], max_x + bbox_w * margin)
                max_y = min(aligned_face.shape[0], max_y + bbox_h * margin)
                
                # Crop to aligned landmarks
                aligned_face = aligned_face[int(min_y):int(max_y), int(min_x):int(max_x)]
            
            return aligned_face
            
        except Exception as e:
            logger.error(f"Error extracting face region: {str(e)}")
            return image[y1:y2, x1:x2]

    def embed_faces_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Process faces in batches with alignment"""
        try:
            if not face_images:
                return []

            # Preprocess and align faces
            preprocessed = []
            for face in face_images:
                # Align face
                aligned_face, _ = self._align_face(face)
                
                # Resize and normalize
                face_resized = cv2.resize(aligned_face, (160, 160))
                face_pixels = face_resized.astype('float32')
                mean, std = face_pixels.mean(), face_pixels.std()
                face_pixels = (face_pixels - mean) / std
                preprocessed.append(np.expand_dims(face_pixels, axis=0))

            # Process in batches
            embeddings = []
            for i in range(0, len(preprocessed), self.batch_size):
                batch = preprocessed[i:i + self.batch_size]
                batch = np.vstack(batch)
                batch_embeddings = self.model.predict(batch, verbose=0)
                embeddings.extend(batch_embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            return []

    def get_unique_people(self) -> List[Dict[str, Any]]:
        """Get unique people with improved cluster verification"""
        try:
            if not self.face_signatures:
                return []

            # Get embeddings
            embeddings = np.array([face.embedding for face in self.face_signatures])

            # Initial clustering
            clustering = DBSCAN(
                eps=0.4,
                min_samples=2,
                metric='cosine',
                n_jobs=-1
            ).fit(embeddings)

            # Group faces
            initial_groups = defaultdict(list)
            for face, label in zip(self.face_signatures, clustering.labels_):
                if label != -1:
                    initial_groups[label].append(face)

            # Verify and refine clusters
            verified_groups = []
            for group in initial_groups.values():
                if len(group) < 2:
                    continue

                # Find centroid face (most confident)
                centroid = max(group, key=lambda x: float(x.confidence))
                centroid_embedding = centroid.embedding

                # Verify each face against centroid
                verified_faces = []
                for face in group:
                    # Calculate cosine similarity
                    similarity = 1 - cosine(centroid_embedding, face.embedding)
                    
                    if similarity >= self.verification_threshold:
                        verified_faces.append(face)
                    else:
                        logger.info(f"Removing face from cluster: similarity {similarity:.3f}")

                if len(verified_faces) >= 2:
                    verified_groups.append(verified_faces)

            # Format output
            unique_people = []
            for idx, group in enumerate(verified_groups):
                representative = max(group, key=lambda x: float(x.confidence))
                avg_confidence = float(np.mean([face.confidence for face in group]))

                unique_people.append({
                    "person_id": idx,
                    "instances_count": len(group),
                    "confidence_score": avg_confidence,
                    "representative_face": {
                        "filename": representative.filename,
                        "face_image": representative.face_image,
                        "confidence": float(representative.confidence),
                        "box": [float(x) for x in representative.box]
                    },
                    "appearances": [{
                        "filename": face.filename,
                        "face_image": face.face_image,
                        "confidence": float(face.confidence),
                        "box": [float(x) for x in face.box]
                    } for face in group]
                })

            logger.info(f"Found {len(unique_people)} unique people after verification")
            return unique_people

        except Exception as e:
            logger.error(f"Error in get_unique_people: {str(e)}")
            logger.exception("Full traceback:")
            return []

def encode_image(img, quality=95):
    """Encode image with higher quality"""
    try:
        if img is None:
            logger.error("Received None image in encode_image")
            return None
            
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Use higher quality JPEG encoding
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', img, encode_param)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in encode_image: {str(e)}")
        return None

def create_high_quality_thumbnail(img_pil, max_size=400):  # Increased from 200
    """Create a high quality thumbnail"""
    try:
        # Calculate new dimensions maintaining aspect ratio
        width, height = img_pil.size
        ratio = min(max_size/width, max_size/height)
        new_size = (int(width * ratio), int(height * ratio))
        
        # Use high quality downsampling
        thumbnail = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        return thumbnail
    except Exception as e:
        logger.error(f"Error creating thumbnail: {str(e)}")
        return None

def extract_metadata(image_path, img_pil):
    """Extract comprehensive metadata from image"""
    try:
        metadata = {
            "basic": {
                "filename": os.path.basename(image_path),
                "file_size": f"{os.path.getsize(image_path) / 1024:.1f} KB",
                "format": img_pil.format,
                "mode": img_pil.mode,
                "dimensions": f"{img_pil.width}x{img_pil.height}",
            },
            "exif": {},
            "gps": {}
        }

        # Get EXIF data using exifread for detailed info
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
        # Extract basic EXIF data
        if tags:
            # Camera details
            if 'Image Make' in tags:
                metadata["exif"]["camera_make"] = str(tags['Image Make'])
            if 'Image Model' in tags:
                metadata["exif"]["camera_model"] = str(tags['Image Model'])
            
            # Capture details
            if 'EXIF DateTimeOriginal' in tags:
                try:
                    date_str = str(tags['EXIF DateTimeOriginal'])
                    date_obj = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                    metadata["exif"]["date_taken"] = date_obj.isoformat()
                except:
                    metadata["exif"]["date_taken"] = str(tags['EXIF DateTimeOriginal'])
            
            # Camera settings
            if 'EXIF ExposureTime' in tags:
                metadata["exif"]["exposure_time"] = str(tags['EXIF ExposureTime'])
            if 'EXIF FNumber' in tags:
                metadata["exif"]["f_stop"] = str(tags['EXIF FNumber'])
            if 'EXIF ISOSpeedRatings' in tags:
                metadata["exif"]["iso"] = str(tags['EXIF ISOSpeedRatings'])
            if 'EXIF FocalLength' in tags:
                metadata["exif"]["focal_length"] = str(tags['EXIF FocalLength'])

            # GPS Data
            gps_tags = {key: tags[key] for key in tags.keys() if key.startswith('GPS')}
            if gps_tags:
                try:
                    if all(tag in tags for tag in ['GPS GPSLatitude', 'GPS GPSLatitudeRef', 
                                                  'GPS GPSLongitude', 'GPS GPSLongitudeRef']):
                        lat = convert_to_degrees(tags['GPS GPSLatitude'])
                        lon = convert_to_degrees(tags['GPS GPSLongitude'])
                        
                        if tags['GPS GPSLatitudeRef'].values[0] != 'N':
                            lat = -lat
                        if tags['GPS GPSLongitudeRef'].values[0] != 'E':
                            lon = -lon
                            
                        metadata["gps"] = {
                            "latitude": f"{lat:.6f}",
                            "longitude": f"{lon:.6f}"
                        }
                        
                        if 'GPS GPSAltitude' in tags:
                            altitude = float(str(tags['GPS GPSAltitude']))
                            metadata["gps"]["altitude"] = f"{altitude:.1f}m"
                except:
                    metadata["gps"] = {"error": "Could not parse GPS data"}

        # Additional PIL metadata
        try:
            exif = img_pil._getexif()
            if exif:
                for tag_id in exif:
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    data = exif.get(tag_id)
                    if isinstance(data, bytes):
                        data = data.decode(errors='ignore')
                    metadata["exif"][tag.lower()] = str(data)
        except:
            pass

        # Software/processing info
        if 'Image Software' in tags:
            metadata["exif"]["software"] = str(tags['Image Software'])
            
        # Copyright information
        if 'Image Copyright' in tags:
            metadata["exif"]["copyright"] = str(tags['Image Copyright'])

        return metadata
    except Exception as e:
        logging.error(f"Error extracting metadata: {str(e)}")
        return {"error": "Failed to extract metadata"}

def convert_to_degrees(gps_coords):
    """Convert GPS coordinates to degrees"""
    d = float(gps_coords.values[0].num) / float(gps_coords.values[0].den)
    m = float(gps_coords.values[1].num) / float(gps_coords.values[1].den)
    s = float(gps_coords.values[2].num) / float(gps_coords.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

def is_valid_detection(box, prob, img_shape, min_confidence=0.9):
    """Optimized validation of face detections"""
    try:
        if prob < min_confidence:
            return False
            
        x1, y1, x2, y2 = [int(b) for b in box]
        w = x2 - x1
        h = y2 - y1
        img_h, img_w = img_shape[:2]
        
        # Basic size check
        if w < 20 or h < 20:
            return False
            
        # Simplified aspect ratio check
        aspect_ratio = w / h
        if not (0.5 <= aspect_ratio <= 2.0):
            return False
            
        # Quick boundary check
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error in face validation: {str(e)}")
        return False

def get_face_region(img, box, padding_factor=0.2):
    """Get face region with proper size handling"""
    try:
        height, width = img.shape[:2]
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Calculate padding
        w = x2 - x1
        h = y2 - y1
        padding = int(max(w, h) * padding_factor)
        
        # Apply padding with boundary checks
        y1_new = max(0, y1 - padding)
        y2_new = min(height, y2 + padding)
        x1_new = max(0, x1 - padding)
        x2_new = min(width, x2 + padding)
        
        face_region = img[y1_new:y2_new, x1_new:x2_new]
        
        # Ensure minimum size
        if face_region.shape[0] < 160 or face_region.shape[1] < 160:
            face_region = cv2.resize(face_region, (160, 160), interpolation=cv2.INTER_LANCZOS4)
            
        return face_region
        
    except Exception as e:
        logger.error(f"Error in get_face_region: {str(e)}")
        return None

def compute_quick_iou(box1, box2):
    """Simplified and faster IOU calculation"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection)

@torch.no_grad()
async def process_image(image_path: Path, filename: str) -> Optional[Dict]:
    try:
        # Load and preprocess image
        img_pil = Image.open(image_path)
        img_pil = ImageOps.exif_transpose(img_pil)
        
        # Create thumbnail
        thumbnail = create_high_quality_thumbnail(img_pil)
        if thumbnail is None:
            return None
            
        # Convert to RGB for processing
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        # Extract metadata
        metadata = extract_metadata(image_path, img_pil)
        
        # Convert to numpy array for face detection
        img = np.array(img_pil)
        
        # Detect faces
        boxes, probs = detector.detect(img_pil)
        
        face_images = []
        valid_faces = []
        embeddings = []
        
        if boxes is not None:
            used_regions = set()
            
            for idx, (box, prob) in enumerate(zip(boxes, probs)):
                if is_valid_detection(box, prob, img.shape):
                    box_tuple = tuple(map(int, box))
                    
                    if not any(compute_quick_iou(box_tuple, used) > 0.3 for used in used_regions):
                        face_img = get_face_region(img, box)
                        if face_img is not None:
                            used_regions.add(box_tuple)
                            
                            # Get embedding for the face
                            try:
                                face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float()
                                face_tensor = face_tensor.unsqueeze(0)
                                face_tensor = (face_tensor - 127.5) / 128.0
                                
                                if torch.cuda.is_available():
                                    face_tensor = face_tensor.cuda()
                                
                                with torch.no_grad():
                                    embedding = resnet(face_tensor).cpu().numpy()[0]
                                
                                valid_faces.append({
                                    "box": box.tolist(),
                                    "confidence": float(prob),
                                    "embedding": embedding.tolist()  # Convert to list for JSON
                                })
                                
                                # Create high quality face thumbnail
                                face_pil = Image.fromarray(face_img)
                                face_thumb = create_high_quality_thumbnail(face_pil, max_size=150)
                                if face_thumb:
                                    face_images.append(encode_image(np.array(face_thumb), quality=95))
                                    embeddings.append(embedding)
                                
                            except Exception as e:
                                logger.error(f"Error processing face {idx} in {filename}: {str(e)}")
                                continue
        
        return {
            "filename": filename,
            "thumbnail": encode_image(np.array(thumbnail), quality=95),
            "faces": valid_faces,
            "face_images": face_images,
            "metadata": metadata,
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return None

@app.get("/analyze-folder")
async def analyze_folder():
    async def generate():
        try:
            folder_path = Path("./images")
            image_files = [
                f for f in folder_path.iterdir()
                if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
            ]
            
            face_clusterer = RobustFaceClusterer()
            
            for img_path in image_files:
                result = await process_image(img_path, img_path.name)
                if result and not result.get('error'):
                    for idx, (face_data, face_img) in enumerate(zip(
                        result.get('faces', []),
                        result.get('face_images', [])
                    )):
                        try:
                            # Convert base64 to image
                            img_bytes = base64.b64decode(face_img)
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            face_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            
                            face_clusterer.face_signatures.append(FaceSignature(
                                embedding=face_data['embedding'],
                                filename=img_path.name,
                                face_image=face_img,
                                confidence=face_data['confidence'],
                                box=face_data['box']
                            ))
                        except Exception as e:
                            logger.error(f"Error processing face: {str(e)}")
                    
                    yield json.dumps({
                        "type": "processing",
                        "result": result
                    }, ensure_ascii=False).strip() + "\n"
            
            unique_people = face_clusterer.get_unique_people()
            
            yield json.dumps({
                "type": "unique_people",
                "people": unique_people
            }, ensure_ascii=False).strip() + "\n"
            
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )
