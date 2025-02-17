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
import face_recognition  # Add this to requirements.txt

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
class FaceInstance:
    filename: str
    face_index: int
    box: List[float]
    confidence: float
    embedding: np.ndarray
    face_image: str  # base64 encoded image

@dataclass
class FacialFeatures:
    landmarks: Dict[str, List[float]]  # Facial landmarks
    pose: Dict[str, float]  # Head pose estimation
    attributes: Dict[str, float]  # Age, gender, expression etc.

class EnhancedFaceClusterer:
    def __init__(
        self,
        min_face_size: int = 160,
        min_confidence: float = 0.92,
        similarity_threshold: float = 0.7,
        min_cluster_size: int = 2
    ):
        self.min_face_size = min_face_size
        self.min_confidence = min_confidence
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.face_instances = []
        self.embeddings = []
        
    def _extract_facial_features(self, face_img: np.ndarray) -> Optional[FacialFeatures]:
        """Extract additional facial features using face_recognition"""
        try:
            # Convert to RGB if needed
            if len(face_img.shape) == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            elif face_img.shape[2] == 4:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
            
            # Get face landmarks
            face_landmarks = face_recognition.face_landmarks(face_img)
            if not face_landmarks:
                return None
                
            landmarks = face_landmarks[0]
            
            # Calculate basic pose estimation from landmarks
            left_eye = np.mean(landmarks['left_eye'], axis=0)
            right_eye = np.mean(landmarks['right_eye'], axis=0)
            nose_tip = landmarks['nose_bridge'][-1]
            
            # Calculate head pose
            eye_angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            
            # Estimate face orientation
            face_direction = np.array(nose_tip) - np.mean([left_eye, right_eye], axis=0)
            face_angle = np.degrees(np.arctan2(face_direction[1], face_direction[0]))
            
            return FacialFeatures(
                landmarks=landmarks,
                pose={
                    'eye_angle': float(eye_angle),
                    'face_angle': float(face_angle)
                },
                attributes={}  # Placeholder for additional attributes
            )
            
        except Exception as e:
            logger.error(f"Error extracting facial features: {str(e)}")
            return None

    def _compute_feature_similarity(
        self,
        features1: FacialFeatures,
        features2: FacialFeatures
    ) -> float:
        """Compute similarity based on facial features"""
        try:
            # Compare landmark configurations
            landmark_similarity = 0
            for key in features1.landmarks.keys():
                points1 = np.array(features1.landmarks[key])
                points2 = np.array(features2.landmarks[key])
                
                # Normalize coordinates
                points1 = (points1 - np.mean(points1, axis=0)) / np.std(points1)
                points2 = (points2 - np.mean(points2, axis=0)) / np.std(points2)
                
                # Compute similarity
                landmark_similarity += 1 - np.mean(np.abs(points1 - points2))
                
            landmark_similarity /= len(features1.landmarks)
            
            # Compare pose angles
            pose_similarity = 1 - (
                abs(features1.pose['eye_angle'] - features2.pose['eye_angle']) / 180 +
                abs(features1.pose['face_angle'] - features2.pose['face_angle']) / 180
            ) / 2
            
            return 0.7 * landmark_similarity + 0.3 * pose_similarity
            
        except Exception as e:
            logger.error(f"Error computing feature similarity: {str(e)}")
            return 0.0

    def _compute_hybrid_similarity(
        self,
        idx1: int,
        idx2: int,
        embedding_similarities: np.ndarray
    ) -> float:
        """Compute hybrid similarity using both embeddings and facial features"""
        try:
            # Get base similarity from embeddings
            emb_similarity = embedding_similarities[idx1, idx2]
            
            # Get facial features similarity
            face1 = self.face_instances[idx1]
            face2 = self.face_instances[idx2]
            
            if hasattr(face1, 'facial_features') and hasattr(face2, 'facial_features'):
                feature_similarity = self._compute_feature_similarity(
                    face1.facial_features,
                    face2.facial_features
                )
            else:
                feature_similarity = 0.5  # Neutral if features not available
            
            # Combine similarities with weights
            return 0.6 * emb_similarity + 0.4 * feature_similarity
            
        except Exception as e:
            logger.error(f"Error computing hybrid similarity: {str(e)}")
            return 0.0

    def _hierarchical_clustering(
        self,
        similarity_matrix: np.ndarray
    ) -> np.ndarray:
        """Perform hierarchical clustering with dynamic threshold"""
        try:
            # Convert similarity to distance
            distance_matrix = 1 - similarity_matrix
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(
                distance_matrix[np.triu_indices(len(distance_matrix), k=1)],
                method='complete'
            )
            
            # Find optimal number of clusters using elbow method
            last = linkage_matrix[-10:, 2]
            acceleration = np.diff(last, 2)
            k = acceleration.argmax() + 2
            k = max(k, self.min_cluster_size)
            
            # Get cluster labels
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            return labels - 1  # Convert to 0-based indexing
            
        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {str(e)}")
            return np.array([-1] * len(similarity_matrix))

    def _compute_cluster_quality(
        self,
        cluster_faces: List[FaceInstance],
        similarity_matrix: np.ndarray
    ) -> float:
        """Compute the quality score for a cluster"""
        try:
            if len(cluster_faces) < 2:
                return 0.0
            
            # Get indices of faces in this cluster
            face_indices = [
                i for i, face in enumerate(self.face_instances)
                if face in cluster_faces
            ]
            
            # Extract similarity submatrix for this cluster
            cluster_similarities = similarity_matrix[np.ix_(face_indices, face_indices)]
            
            # Compute various quality metrics
            # 1. Average pairwise similarity
            np.fill_diagonal(cluster_similarities, 0)  # Exclude self-similarities
            avg_similarity = np.mean(cluster_similarities)
            
            # 2. Minimum pairwise similarity
            min_similarity = np.min(cluster_similarities[cluster_similarities > 0])
            
            # 3. Consistency score (how many pairs are above threshold)
            pairs_above_threshold = np.sum(cluster_similarities > self.similarity_threshold)
            total_pairs = (len(cluster_faces) * (len(cluster_faces) - 1)) / 2
            consistency_score = pairs_above_threshold / total_pairs
            
            # 4. Average face confidence
            avg_confidence = np.mean([face.confidence for face in cluster_faces])
            
            # 5. Facial features consistency (if available)
            feature_consistency = 0.0
            feature_count = 0
            
            for face in cluster_faces:
                if hasattr(face, 'facial_features'):
                    feature_count += 1
            
            if feature_count >= 2:
                feature_similarities = []
                for i, face1 in enumerate(cluster_faces[:-1]):
                    for face2 in cluster_faces[i+1:]:
                        if hasattr(face1, 'facial_features') and hasattr(face2, 'facial_features'):
                            sim = self._compute_feature_similarity(
                                face1.facial_features,
                                face2.facial_features
                            )
                            feature_similarities.append(sim)
            
            if feature_similarities:
                feature_consistency = np.mean(feature_similarities)
            
            # Combine all metrics with weights
            quality_score = (
                0.3 * avg_similarity +
                0.2 * min_similarity +
                0.2 * consistency_score +
                0.15 * avg_confidence +
                0.15 * feature_consistency
            )
            
            return float(quality_score)
            
        except Exception as e:
            logger.error(f"Error computing cluster quality: {str(e)}")
            return 0.0

    def _select_representative_face(self, cluster_faces: List[FaceInstance]) -> FaceInstance:
        """Select the best representative face from a cluster using multiple criteria"""
        try:
            if not cluster_faces:
                return None
            
            face_scores = []
            for face in cluster_faces:
                score = 0.0
                
                # 1. Base confidence score (30%)
                score += 0.3 * face.confidence
                
                # 2. Face position and size score (25%)
                box = face.box
                width = box[2] - box[0]
                height = box[3] - box[1]
                size = width * height
                
                # Prefer centered faces
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                center_offset = abs(0.5 - center_x) + abs(0.5 - center_y)
                position_score = 1 - (center_offset / 2)
                
                # Prefer larger faces, but not too large
                size_score = min(1.0, size / (self.min_face_size ** 2))
                
                score += 0.25 * (0.6 * position_score + 0.4 * size_score)
                
                # 3. Facial features quality score (25%)
                if hasattr(face, 'facial_features'):
                    feature_score = 0.0
                    
                    # Check if eyes are open and visible
                    if 'left_eye' in face.facial_features.landmarks and 'right_eye' in face.facial_features.landmarks:
                        left_eye = np.array(face.facial_features.landmarks['left_eye'])
                        right_eye = np.array(face.facial_features.landmarks['right_eye'])
                        
                        # Check eye aspect ratio
                        left_ear = self._eye_aspect_ratio(left_eye)
                        right_ear = self._eye_aspect_ratio(right_eye)
                        eye_score = min(1.0, (left_ear + right_ear) / 0.5)
                        feature_score += 0.4 * eye_score
                    
                    # Check face angle
                    pose = face.facial_features.pose
                    angle_penalty = 1.0 - (abs(pose['face_angle']) / 45.0)  # Penalize angles > 45 degrees
                    feature_score += 0.6 * max(0, angle_penalty)
                    
                    score += 0.25 * feature_score
                
                # 4. Similarity to cluster center score (20%)
                face_idx = self.face_instances.index(face)
                cluster_indices = [self.face_instances.index(f) for f in cluster_faces]
                similarities = [
                    1 - cosine(self.embeddings[face_idx], self.embeddings[i])
                    for i in cluster_indices
                ]
                center_similarity = np.mean(similarities)
                score += 0.2 * center_similarity
                
                face_scores.append(score)
            
            # Select face with highest score
            best_idx = np.argmax(face_scores)
            return cluster_faces[best_idx]
            
        except Exception as e:
            logger.error(f"Error selecting representative face: {str(e)}")
            # Fallback to highest confidence face
            return max(cluster_faces, key=lambda x: x.confidence)

    def _eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """Calculate eye aspect ratio to detect open eyes"""
        try:
            # Compute the euclidean distances
            vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
            vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
            horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
            
            # Compute eye aspect ratio
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return float(ear)
            
        except Exception as e:
            logger.error(f"Error calculating eye aspect ratio: {str(e)}")
            return 0.0

    def cluster_faces(self) -> List[Dict[str, Any]]:
        """Perform enhanced face clustering"""
        if len(self.face_instances) < self.min_cluster_size:
            return []
            
        try:
            # Compute embedding similarities
            embeddings_array = np.array(self.embeddings)
            embedding_similarities = 1 - np.array([
                [cosine(emb1, emb2) for emb2 in embeddings_array]
                for emb1 in embeddings_array
            ])
            
            # Compute hybrid similarity matrix
            n_faces = len(self.face_instances)
            hybrid_similarities = np.zeros((n_faces, n_faces))
            
            for i in range(n_faces):
                for j in range(i + 1, n_faces):
                    similarity = self._compute_hybrid_similarity(
                        i, j, embedding_similarities
                    )
                    hybrid_similarities[i, j] = similarity
                    hybrid_similarities[j, i] = similarity
            
            # Perform hierarchical clustering
            cluster_labels = self._hierarchical_clustering(hybrid_similarities)
            
            # Group faces by cluster
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label >= 0:
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(self.face_instances[idx])
            
            # Process clusters
            unique_people = []
            for cluster_id, cluster_faces in clusters.items():
                if len(cluster_faces) >= self.min_cluster_size:
                    # Compute cluster quality
                    quality_score = self._compute_cluster_quality(
                        cluster_faces,
                        hybrid_similarities
                    )
                    
                    if quality_score >= 0.75:  # Higher threshold for better precision
                        representative_face = self._select_representative_face(cluster_faces)
                        
                        unique_people.append({
                            "person_id": int(cluster_id),
                            "instances_count": len(cluster_faces),
                            "confidence_score": float(quality_score),
                            "representative_face": {
                                "filename": representative_face.filename,
                                "face_image": representative_face.face_image,
                                "confidence": float(representative_face.confidence)
                            },
                            "appearances": [
                                {
                                    "filename": face.filename,
                                    "face_image": face.face_image,
                                    "confidence": float(face.confidence),
                                    "box": [float(x) for x in face.box]
                                }
                                for face in sorted(
                                    cluster_faces,
                                    key=lambda x: x.confidence,
                                    reverse=True
                                )
                            ]
                        })
            
            # Sort by quality and size
            unique_people.sort(
                key=lambda x: (x["confidence_score"], x["instances_count"]),
                reverse=True
            )
            
            return unique_people
            
        except Exception as e:
            logger.error(f"Error in cluster_faces: {str(e)}")
            return []

    def add_face(self, face_instance: FaceInstance) -> None:
        """Add a face for clustering with enhanced features"""
        if face_instance.confidence >= self.min_confidence:
            try:
                # Convert base64 to image for feature extraction
                img_bytes = base64.b64decode(face_instance.face_image)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                face_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Extract facial features
                facial_features = self._extract_facial_features(face_img)
                if facial_features:
                    face_instance.facial_features = facial_features
                    self.face_instances.append(face_instance)
                    self.embeddings.append(face_instance.embedding)
                    
            except Exception as e:
                logger.error(f"Error adding face: {str(e)}")

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

def is_valid_detection(box, prob, img_shape, min_confidence=0.85):
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
            
            face_clusterer = EnhancedFaceClusterer(
                min_face_size=160,
                min_confidence=0.92,
                similarity_threshold=0.7,
                min_cluster_size=2
            )
            
            # Process images
            for img_path in image_files:
                result = await process_image(img_path, img_path.name)
                if result and not result.get('error'):
                    for idx, (face_data, face_img) in enumerate(zip(
                        result.get('faces', []),
                        result.get('face_images', [])
                    )):
                        try:
                            face_instance = FaceInstance(
                                filename=img_path.name,
                                face_index=idx,
                                box=face_data['box'],
                                confidence=face_data['confidence'],
                                embedding=np.array(face_data['embedding']),
                                face_image=face_img
                            )
                            face_clusterer.add_face(face_instance)
                        except Exception as e:
                            logger.error(f"Error creating face instance: {str(e)}")
                    
                    yield json.dumps({
                        "type": "processing",
                        "result": result
                    }, ensure_ascii=False).strip() + "\n"
            
            # Get unique people
            unique_people = face_clusterer.cluster_faces()
            
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
