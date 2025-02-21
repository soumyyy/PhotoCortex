import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from insightface.app import FaceAnalysis
from dataclasses import dataclass, asdict
import warnings
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import torch
import torchvision.models as models
from torchvision import transforms
from ultralytics import YOLO

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class FaceSignature:
    embedding: np.ndarray
    filename: str
    face_image: str
    confidence: float
    box: List[float]
    landmarks: np.ndarray

    def __post_init__(self):
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


class FaceClusterer:
    def __init__(self):
        """Initialize with suppressed stdout."""
        self.min_confidence = 0.6
        logger.info("Initializing FaceClusterer...")
        
        # Initialize YOLO first
        self.yolo_model = None
        try:
            model_path = './models/yolo_v8s/yolov8s.pt'
            if os.path.exists(model_path):
                # Suppress YOLO initialization messages
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.yolo_model = YOLO(model_path, verbose=False)
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {str(e)}")
        
        # Initialize other models
        with redirect_stdout(io.StringIO()):
            self.app = FaceAnalysis(
                name='buffalo_l',
                root=".models/insightface_model",
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
            self.load_indoor_outdoor_model()

        logger.info("All models initialized.")

    def load_indoor_outdoor_model(self):
        """Load Places365 model for scene classification."""
        try:
            # Load the model architecture (ResNet50)
            self.scene_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            num_ftrs = self.scene_model.fc.in_features
            self.scene_model.fc = torch.nn.Linear(num_ftrs, 365)  # Places365 has 365 categories
            
            # Load the Places365 weights
            model_path = './models/indoor_outdoor_classifier/resnet50_places365.pth.tar'
            if not os.path.exists(model_path):
                logger.warning("Places365 model not found. Scene classification will be disabled.")
                self.scene_model = None
                return
                
            # Load Places365 checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            
            # Remove the 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load all weights including fc layer
            model_dict = self.scene_model.state_dict()
            model_dict.update(state_dict)
            self.scene_model.load_state_dict(model_dict)
            
            # Verify model configuration
            logger.info(f"FC layer output features: {self.scene_model.fc.out_features}")
            logger.info(f"Sample model layers: {list(state_dict.keys())[:5]}")
            
            # Define broader ranges for indoor/outdoor in Places365
            # Expanding ranges to catch more categories
            self.indoor_ranges = [(0, 150)]  # First 150 categories are typically indoor
            self.outdoor_ranges = [(250, 365)]  # Last 115 categories are typically outdoor
            
            # These are just for logging
            self.categories = [f"category_{i}" for i in range(365)]
            self.indoor_indices = [i for start, end in self.indoor_ranges for i in range(start, end)]
            self.outdoor_indices = [i for start, end in self.outdoor_ranges for i in range(start, end)]
            
            logger.info(f"Loaded {len(self.categories)} categories")
            logger.info(f"Indoor categories: {len(self.indoor_indices)}, Outdoor categories: {len(self.outdoor_indices)}")
            
            # Define image preprocessing pipeline
            self.scene_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),  # Resize shorter side to 256
                transforms.CenterCrop(224),  # Center crop to 224x224
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Set model to eval mode
            self.scene_model.eval()
            
            # Define image transforms
            self.scene_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            logger.info("Indoor/outdoor classification model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading indoor/outdoor model: {e}")
            self.scene_model = None

    def predict_indoor_outdoor(self, image_bytes):
        """Predict whether an image is indoor or outdoor using Places365 categories."""
        if self.scene_model is None:
            return {
                "scene_type": "unknown",
                "confidence": 0.0,
                "probabilities": {
                    "indoor": 0.0,
                    "outdoor": 0.0
                }
            }
            
        try:
            # Convert image bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Prepare image for model
            input_tensor = self.scene_transforms(img)
            input_batch = input_tensor.unsqueeze(0)
            
            # Get model prediction
            with torch.no_grad():
                output = self.scene_model(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Get predictions but don't log them
                top5_values, top5_indices = torch.topk(probabilities, 5)
                
                # Map Places365 predictions to our categories
                category_probs = {}
                for i in range(len(probabilities)):
                    prob = probabilities[i].item()
                    if prob > 0.01:  # Only consider probabilities > 1%
                        category_probs[i] = prob
                
                # Calculate weighted indoor/outdoor probabilities
                indoor_prob = 0.0
                outdoor_prob = 0.0
                total_weight = 0.0
                
                # Common indoor/outdoor keywords in Places365
                indoor_keywords = ['room', 'indoor', 'inside', 'interior', 'bedroom', 'kitchen', 'office', 'bathroom']
                outdoor_keywords = ['outdoor', 'outside', 'field', 'sky', 'mountain', 'forest', 'street', 'beach']
                
                # Process each prediction
                for idx, prob in category_probs.items():
                    # Get the raw Places365 category name (even if we don't have it mapped)
                    category_name = f"category_{idx}"
                    
                    # Determine if it's indoor/outdoor based on index patterns in Places365
                    # Using broader ranges and higher weights
                    if idx < 150:  # Indoor categories
                        weight = 1.0 if idx < 100 else 0.8  # Higher weight for more confident categories
                        indoor_prob += prob * weight
                        total_weight += weight
                    elif idx > 250:  # Outdoor categories
                        weight = 1.0 if idx > 300 else 0.8  # Higher weight for more confident categories
                        outdoor_prob += prob * weight
                        total_weight += weight
                
                # Convert to percentages and normalize
                if total_weight > 0:
                    indoor_prob = (indoor_prob / total_weight) * 100
                    outdoor_prob = (outdoor_prob / total_weight) * 100
                else:
                    indoor_prob = outdoor_prob = 0  # No confident predictions
                
                # Always show the most likely prediction
                if indoor_prob > outdoor_prob:
                    scene_type = "indoor"
                    confidence = indoor_prob
                else:
                    scene_type = "outdoor"
                    confidence = outdoor_prob
                
            prediction = {
                "scene_type": scene_type
            }
            
            return prediction
        except Exception as e:
            logger.error(f"Error in scene prediction: {e}")
            return {"error": str(e)}

    def _validate_face(self, face_img: np.ndarray) -> bool:
        """Validate face aspect ratio."""
        try:
            h, w = face_img.shape[:2]
            return (h >= 32) and (w >= 32) and (0.7 <= w / h <= 1.3)
        except Exception as e:
            logger.error(f"Face validation error: {e}")
            return False

    def _encode_face_image(self, image: np.ndarray, bbox) -> str:
        """Extract, pad, and encode face region *without* converting to RGB."""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]

            padding_x = int((x2 - x1) * 0.3)
            padding_y = int((y2 - y1) * 0.3)
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(w, x2 + padding_x)
            y2 = min(h, y2 + padding_y)

            if x2 <= x1 or y2 <= y1:
                return ""
            face_img = image[y1:y2, x1:x2].copy()
            if not self._validate_face(face_img):
                return ""

            # NO cvtColor here → keep original BGR
            face_img = cv2.resize(face_img, (224, 224))

            success, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                return ""
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            logger.error(f"Error encoding face: {e}")
            return ""

    def extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract comprehensive metadata from image."""
        from PIL import Image
        try:
            with Image.open(image_path) as img:
                file_stats = os.stat(image_path)
                metadata = {
                    "basic": {
                        "filename": image_path.name,
                        "file_size": f"{file_stats.st_size / (1024 * 1024):.2f} MB",
                        "dimensions": f"{img.width}x{img.height}",
                        "format": img.format,
                        "mode": img.mode,
                        "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    },
                    "exif": {},
                    "gps": {}
                }

                exif = img.getexif()
                if exif:
                    from PIL.ExifTags import TAGS, GPSTAGS
                    for tag_id in exif:
                        tag = TAGS.get(tag_id, tag_id)
                        data = exif.get(tag_id)
                        if isinstance(data, bytes):
                            continue
                        try:
                            if isinstance(data, tuple):
                                data = data[0] / data[1]
                            metadata["exif"][tag.lower()] = str(data)
                        except:
                            continue

                    if hasattr(exif, '_getexif'):
                        labeled = {}
                        for key, val in exif._getexif().items():
                            labeled[TAGS.get(key)] = val

                        if 'GPSInfo' in labeled:
                            gps_data = {}
                            for t in labeled['GPSInfo'].keys():
                                sub_decoded = GPSTAGS.get(t, t)
                                gps_data[sub_decoded] = labeled['GPSInfo'][t]

                            if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                                try:
                                    lat = self._convert_to_degrees(gps_data['GPSLatitude'])
                                    lon = self._convert_to_degrees(gps_data['GPSLongitude'])

                                    if gps_data['GPSLatitudeRef'] != 'N':
                                        lat = -lat
                                    if gps_data['GPSLongitudeRef'] != 'E':
                                        lon = -lon

                                    metadata["gps"] = {
                                        "latitude": f"{lat:.6f}",
                                        "longitude": f"{lon:.6f}"
                                    }

                                    if 'GPSAltitude' in gps_data:
                                        alt = float(gps_data['GPSAltitude'].real)
                                        metadata["gps"]["altitude"] = f"{alt:.1f}m"
                                except:
                                    pass
                return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"error": f"Failed to extract metadata: {str(e)}"}

    def _convert_to_degrees(self, value):
        """Helper to convert GPS coordinates to degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)

    def process_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single image using InsightFace and YOLOv8, keep BGR for output."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read: {image_path.name}")
                return None

            metadata = self.extract_metadata(image_path)

            # Read image bytes for indoor/outdoor classification
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Get indoor/outdoor classification
            scene_info = self.predict_indoor_outdoor(image_bytes)

            # Possibly resize if very large
            original_img = image.copy()
            h, w = image.shape[:2]
            max_size = 1920
            resized_img = original_img
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                resized_img = cv2.resize(image, (int(w * scale), int(h * scale)))

            # Detect faces
            faces = self.app.get(resized_img)
            results = []
            for face in faces:
                if face.det_score >= self.min_confidence:
                    face_image = self._encode_face_image(resized_img, face.bbox)
                    if face_image:
                        results.append({
                            'bbox': face.bbox.tolist(),
                            'confidence': float(face.det_score),
                            'embedding': face.embedding.tolist(),
                            'filename': str(image_path.name),
                            'face_image': face_image
                        })

            # Run YOLOv8 object detection
            object_detections = set()  # Using set to avoid duplicates
            if self.yolo_model is not None:
                try:
                    # Suppress all YOLO output
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        yolo_results = self.yolo_model(resized_img, verbose=False)
                    for result in yolo_results:
                        boxes = result.boxes
                        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                            class_name = result.names[int(cls)]
                            # Skip person class and only include high confidence detections
                            if class_name != 'person' and float(conf) > 0.3:
                                object_detections.add(class_name)
                    # Convert set to sorted list
                    object_detections = sorted(list(object_detections))
                except Exception as e:
                    logger.error(f"Error in YOLOv8 detection: {e}")

            # Encode the full (original) image in BGR → JPEG
            success, buffer = cv2.imencode('.jpg', original_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                raise Exception("Failed to encode full image")

            full_image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Create info string for frontend
            scene_type = scene_info.get('scene_type', 'unknown')
            objects_str = ', '.join(object_detections) if object_detections else 'none'
            info_str = f"Scene: {scene_type}, Objects: {objects_str}"

            return {
                'filename': str(image_path.name),
                'faces': results,
                'full_image': f"data:image/jpeg;base64,{full_image_base64}",
                'metadata': metadata,
                'scene_classification': {'scene_type': scene_type},
                'object_detections': object_detections,
                'info': info_str
            }
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return None

    def process_image_batch(self, image_paths: List[Path]) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of images."""
        results = []
        for path in image_paths:
            outcome = self.process_image(path)
            if outcome:
                results.append(outcome)
        return results

    def cluster_faces(self, all_faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster faces using DBSCAN."""
        if not all_faces:
            logger.info("No faces to cluster.")
            return []

        if len(all_faces) == 1:
            face = all_faces[0]
            return [{
                "person_id": 0,
                "instances_count": 1,
                "confidence_score": float(face['confidence']),
                "representative_face": {
                    "filename": face['filename'],
                    "face_image": face['face_image'],
                    "confidence": face['confidence'],
                    "box": face['bbox']
                },
                "appearances": [{
                    "filename": face['filename'],
                    "face_image": face['face_image'],
                    "confidence": face['confidence'],
                    "box": face['bbox']
                }]
            }]

        embeddings = normalize(np.array([face['embedding'] for face in all_faces]))
        # Slightly increased eps (0.5) for possible better merging
        clustering = DBSCAN(eps=0.5, min_samples=1, metric='cosine', n_jobs=-1).fit(embeddings)

        clusters = {}
        for face, label in zip(all_faces, clustering.labels_):
            clusters.setdefault(label, []).append(face)

        logger.info(f"Found {len(clusters)} clusters.")
        unique_people = []
        for idx, faces in clusters.items():
            representative = max(faces, key=lambda x: x['confidence'])
            person_data = {
                "person_id": int(idx),
                "instances_count": len(faces),
                "confidence_score": float(np.mean([f['confidence'] for f in faces])),
                "representative_face": {
                    "filename": representative['filename'],
                    "face_image": representative['face_image'],
                    "confidence": representative['confidence'],
                    "box": representative['bbox']
                },
                "appearances": [{
                    "filename": f['filename'],
                    "face_image": f['face_image'],
                    "confidence": f['confidence'],
                    "box": f['bbox']
                } for f in faces]
            }
            unique_people.append(person_data)

        unique_people.sort(key=lambda x: x['instances_count'], reverse=True)
        logger.info(f"Returning {len(unique_people)} unique people.")
        return unique_people


@app.get("/analyze-folder")
async def analyze_folder():
    """Endpoint to analyze the entire folder in batches using multiprocessing."""
    async def generate():
        try:
            logger.info("Starting analysis.")
            clusterer = FaceClusterer()
            folder_path = Path("./images")

            if not folder_path.exists():
                logger.error("Images folder not found.")
                raise FileNotFoundError("Images folder not found.")

            image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

            batch_size = 8
            all_results = []
            
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(clusterer.process_image_batch, image_files[i:i+batch_size])
                    for i in range(0, len(image_files), batch_size)
                ]

                for future in as_completed(futures):
                    batch_results = future.result()
                    for result in batch_results:
                        if result:
                            all_results.append(result)
                            yield json.dumps({
                                "type": "processing",
                                "result": result
                            }) + "\n"

            all_faces = [face for result in all_results for face in result['faces']]
            unique_people = clusterer.cluster_faces(all_faces)
            yield json.dumps({
                "type": "unique_people",
                "people": unique_people
            }) + "\n"

            logger.info("Analysis complete.")

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")