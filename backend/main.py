import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN
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

# Initialize MTCNN with explicit error handling
try:
    # Force CPU usage initially
    device = 'cpu'  # Start with CPU to ensure it works
    logger.info(f"Initializing MTCNN on {device}")
    
    detector = MTCNN(
        keep_all=True,
        device=device,
        selection_method='largest',  # Changed to most reliable method
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,  # Enable post processing
    )
    
    # Test the detector with a small blank image
    test_img = Image.new('RGB', (100, 100))
    test_result = detector.detect(test_img)
    logger.info("MTCNN initialization successful")
    
except Exception as e:
    logger.error(f"Failed to initialize MTCNN: {str(e)}")
    logger.error(f"Python version: {sys.version}")
    logger.error(f"Torch version: {torch.__version__}")
    raise RuntimeError(f"MTCNN initialization failed: {str(e)}")

def encode_image(img):
    try:
        if img is None:
            logger.error("Received None image in encode_image")
            return None
            
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in encode_image: {str(e)}")
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

async def process_image(image_path, filename):
    try:
        logger.info(f"Processing image: {filename}")
        
        # Verify file exists and is readable
        if not os.path.isfile(image_path):
            logger.error(f"File not found: {image_path}")
            return None
            
        # Open and verify image
        img_pil = Image.open(image_path)
        if img_pil is None:
            logger.error(f"Failed to open image: {image_path}")
            return None
            
        # Extract metadata
        metadata = extract_metadata(image_path, img_pil)
        
        img_pil = ImageOps.exif_transpose(img_pil)
        img_pil = img_pil.convert('RGB')
        
        # Convert to numpy array
        img = np.array(img_pil)
        if img is None or img.size == 0:
            logger.error(f"Failed to convert image to array: {filename}")
            return None
            
        # Create thumbnail
        h, w = img.shape[:2]
        scale = 200 / max(h, w)
        thumbnail = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # Detect faces
        try:
            boxes, probs = detector.detect(img_pil)
        except Exception as e:
            logger.error(f"Face detection failed for {filename}: {str(e)}")
            return None
            
        if boxes is None:
            logger.info(f"No faces detected in {filename}")
            return None
            
        face_images = []
        valid_faces = []
        
        for box, prob in zip(boxes, probs):
            try:
                if prob < 0.8:  # Confidence threshold
                    continue
                    
                x1, y1, x2, y2 = [int(b) for b in box]
                face_img = img[y1:y2, x1:x2]
                
                if face_img is None or face_img.size == 0:
                    continue
                    
                # Resize face image
                target_size = 100
                aspect = (x2 - x1) / (y2 - y1)
                if aspect > 1:
                    new_w = target_size
                    new_h = int(target_size / aspect)
                else:
                    new_h = target_size
                    new_w = int(target_size * aspect)
                    
                face_img_resized = cv2.resize(face_img, (new_w, new_h))
                
                valid_faces.append(box.tolist())
                face_images.append(encode_image(face_img_resized))
                
            except Exception as e:
                logger.error(f"Error processing face in {filename}: {str(e)}")
                continue
        
        if not face_images:
            return None
            
        return {
            "filename": filename,
            "thumbnail": encode_image(thumbnail),
            "analysis": {
                "faces_detected": len(valid_faces),
                "face_locations": valid_faces,
                "face_images": face_images,
                "dimensions": {
                    "width": img.shape[1],
                    "height": img.shape[0]
                },
                "confidence": "high"
            },
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return None

@app.get("/analyze-folder")
async def analyze_folder():
    async def generate():
        try:
            folder_path = "./images"
            if not os.path.exists(folder_path):
                logger.error("Images folder not found")
                raise HTTPException(status_code=404, detail="Folder not found")
            
            image_files = [
                f for f in os.listdir(folder_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            logger.info(f"Found {len(image_files)} images to process")
            
            for filename in image_files:
                try:
                    result = await process_image(
                        os.path.join(folder_path, filename), 
                        filename
                    )
                    if result:
                        yield json.dumps(result, ensure_ascii=False).strip() + "\n"
                        await asyncio.sleep(0)
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked"
        }
    )