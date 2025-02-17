import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS requests from your Next.js front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load cascades with error handling
def load_cascade(cascade_name):
    cascade_path = cv2.data.haarcascades + cascade_name
    if not os.path.exists(cascade_path):
        print(f"Warning: {cascade_name} not found")
        return None
    return cv2.CascadeClassifier(cascade_path)

# Initialize cascades
face_cascade_front = load_cascade('haarcascade_frontalface_default.xml')
face_cascade_profile = load_cascade('haarcascade_profileface.xml')
eye_cascade = load_cascade('haarcascade_eye.xml')

if not face_cascade_front or not face_cascade_profile:
    raise RuntimeError("Required cascade classifiers not found")

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def is_valid_face(face_img, gray_face):
    if face_img is None or face_img.size == 0:
        return False
    
    # Only check for eyes if eye cascade is available
    if eye_cascade is not None:
        eyes = eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20)
        )
        # If no eyes detected, might be a false positive
        if len(eyes) == 0:
            return False
    
    # Basic size validation
    min_face_size = 30
    if face_img.shape[0] < min_face_size or face_img.shape[1] < min_face_size:
        return False
    
    # Aspect ratio check
    aspect_ratio = face_img.shape[1] / face_img.shape[0]
    if not (0.5 <= aspect_ratio <= 2.0):
        return False
    
    # Variance check for contrast
    if np.var(gray_face) < 100:
        return False

    return True

def get_face_region(img, x, y, w, h):
    """Get face region with smart padding based on image dimensions"""
    height, width = img.shape[:2]
    
    # Calculate padding (more on top and sides, less on bottom)
    top_pad = int(h * 0.5)      # 50% padding on top for hair
    side_pad = int(w * 0.3)     # 30% padding on sides
    bottom_pad = int(h * 0.2)   # 20% padding on bottom
    
    # Calculate coordinates with padding
    y1 = max(0, y - top_pad)
    y2 = min(height, y + h + bottom_pad)
    x1 = max(0, x - side_pad)
    x2 = min(width, x + w + side_pad)
    
    return img[y1:y2, x1:x2]

@app.get("/")
def read_root():
    return {"message": "Hello from Cortex AI backend!"}

@app.get("/analyze-folder")
async def analyze_folder():
    try:
        folder_path = "./images"
        
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail="Folder not found")
            
        results = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                
                # Read and process the image
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Create thumbnail
                max_size = 200
                h, w = img.shape[:2]
                scale = max_size / max(h, w)
                thumbnail = cv2.resize(img, (int(w * scale), int(h * scale)))
                
                # Enhance image for better face detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                # Detect faces with optimized parameters
                faces_front = face_cascade_front.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    maxSize=(1000, 1000)
                )
                
                faces_profile = face_cascade_profile.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    maxSize=(1000, 1000)
                )
                
                # Combine and filter faces
                all_faces = faces_front.tolist() if len(faces_front) > 0 else []
                all_faces.extend(faces_profile.tolist() if len(faces_profile) > 0 else [])
                filtered_faces = []
                face_images = []
                
                # Extract and validate face regions
                for (x, y, w, h) in all_faces:
                    # Get face region with smart padding
                    face_img = get_face_region(img, x, y, w, h)
                    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    
                    if is_valid_face(face_img, face_gray):
                        filtered_faces.append([x, y, w, h])
                        # Resize maintaining aspect ratio
                        target_size = 100
                        aspect = face_img.shape[1] / face_img.shape[0]
                        if aspect > 1:
                            new_w = target_size
                            new_h = int(target_size / aspect)
                        else:
                            new_h = target_size
                            new_w = int(target_size * aspect)
                        face_img_resized = cv2.resize(face_img, (new_w, new_h))
                        face_images.append(encode_image(face_img_resized))
                
                # Only include images with valid face detections
                if len(filtered_faces) > 0:
                    results.append({
                        "filename": filename,
                        "thumbnail": encode_image(thumbnail),
                        "analysis": {
                            "faces_detected": len(filtered_faces),
                            "face_locations": filtered_faces,
                            "face_images": face_images,
                            "dimensions": {
                                "width": img.shape[1],
                                "height": img.shape[0]
                            },
                            "confidence": "high" if len(filtered_faces) > 0 else "low"
                        }
                    })
                
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))