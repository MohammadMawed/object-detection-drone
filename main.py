#!/usr/bin/env python3
"""
Raspberry Pi Optimized Drone Camera System with Face Recognition and Object Tracking
Optimized for Pi 4/5 with minimal CPU/memory usage
Requires: picamera2, opencv-python, face-recognition, numpy
Install: pip install picamera2 opencv-python face-recognition numpy pillow
"""

import cv2
import numpy as np
import face_recognition
import pickle
import os
import time
import json
from datetime import datetime
from picamera2 import Picamera2
from threading import Thread, Lock
import logging
from collections import deque
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedDroneVisionSystem:
    def __init__(self, face_db_path="face_database.pkl", capture_path="captures/"):
        self.picam2 = Picamera2()
        self.face_db_path = face_db_path
        self.capture_path = capture_path
        self.known_faces = {}
        self.face_lock = Lock()
        
        # Performance optimization settings
        self.frame_width = 320  # Reduced resolution for better performance
        self.frame_height = 240
        self.display_width = 640  # Display scaling
        self.display_height = 480
        self.target_fps = 15  # Reduced FPS for Pi stability
        
        # Processing intervals (process every N frames)
        self.face_detection_interval = 10  # Process faces every 10 frames
        self.motion_detection_interval = 3  # Motion every 3 frames
        self.hud_update_interval = 5       # HUD every 5 frames
        self.frame_count = 0
        
        # Create directories
        os.makedirs(capture_path, exist_ok=True)
        os.makedirs("face_crops", exist_ok=True)
        
        # Load existing face database
        self.load_face_database()
        
        # Object tracking variables (lightweight tracker)
        self.tracker = None
        self.tracking_active = False
        self.track_bbox = None
        
        # Frame buffers for processing
        self.frame_buffer = deque(maxlen=3)
        self.processed_frame = None
        
        # Background subtractor with reduced learning rate
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,  # Disable shadows for performance
            varThreshold=50,      # Increased threshold
            history=100           # Reduced history
        )
        
        # Statistics
        self.stats = {
            'faces_detected': 0,
            'new_faces_saved': 0,
            'objects_tracked': 0,
            'session_start': datetime.now(),
            'frame_drops': 0
        }
        
        # FPS tracking
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        self.setup_camera()
        
    def setup_camera(self):
        """Initialize camera with Pi-optimized settings"""
        try:
            # Optimized configuration for Pi
            config = self.picam2.create_preview_configuration(
                main={
                    "format": 'RGB888',  # Simpler format
                    "size": (self.frame_width, self.frame_height)
                },
                controls={
                    "FrameRate": self.target_fps,
                    "ExposureTime": 10000,  # Fixed exposure for consistent performance
                    "AnalogueGain": 1.0,    # Fixed gain
                    "AeEnable": False,      # Disable auto exposure
                    "AwbEnable": False      # Disable auto white balance
                }
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            # Warm up camera
            time.sleep(1)
            logger.info(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.target_fps}fps")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
    
    def load_face_database(self):
        """Load known faces from pickle file"""
        if os.path.exists(self.face_db_path):
            try:
                with open(self.face_db_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                logger.info(f"Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                logger.error(f"Error loading face database: {e}")
                self.known_faces = {}
        else:
            logger.info("No existing face database found")
    
    def save_face_database(self):
        """Save known faces to pickle file (async)"""
        def save_async():
            try:
                with self.face_lock:
                    with open(self.face_db_path, 'wb') as f:
                        pickle.dump(self.known_faces, f)
                logger.info(f"Face database saved with {len(self.known_faces)} faces")
            except Exception as e:
                logger.error(f"Error saving face database: {e}")
        
        # Save in background thread to not block main loop
        Thread(target=save_async, daemon=True).start()
    
    def add_new_face(self, face_encoding, face_image, person_id):
        """Add a new face to the database (optimized)"""
        with self.face_lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Only save face crop if we have space (prevent storage overflow)
            if len(self.known_faces) < 100:  # Limit to 100 faces
                face_filename = f"face_crops/person_{person_id}_{timestamp}.jpg"
                # Compress image to save space
                cv2.imwrite(face_filename, face_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            else:
                face_filename = None
            
            # Add to database with minimal data
            self.known_faces[person_id] = {
                'encoding': face_encoding.tolist(),  # Convert to list for JSON compatibility
                'first_seen': timestamp,
                'sightings': 1
            }
            
            if face_filename:
                self.known_faces[person_id]['image_path'] = face_filename
            
            self.stats['new_faces_saved'] += 1
            logger.info(f"New face saved as person_{person_id}")
    
    def update_face_sighting(self, person_id):
        """Update sighting count for known face"""
        with self.face_lock:
            if person_id in self.known_faces:
                self.known_faces[person_id]['sightings'] += 1
    
    def process_faces_optimized(self, frame):
        """Optimized face detection with reduced processing"""
        # Scale down frame for face detection to improve speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Use faster HOG model
        face_locations = face_recognition.face_locations(rgb_small, 
                                                        number_of_times_to_upsample=1, 
                                                        model="hog")
        
        if not face_locations:
            return frame, []
        
        # Limit to max 3 faces for performance
        face_locations = face_locations[:3]
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        self.stats['faces_detected'] += len(face_locations)
        
        face_info = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale coordinates back up
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            name = "Unknown"
            confidence = 0
            
            if len(self.known_faces) > 0:
                # Compare with known faces (reduced tolerance for speed)
                known_encodings = [np.array(data['encoding']) for data in self.known_faces.values()]
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.7)
                
                if True in matches:
                    match_index = matches.index(True)
                    person_id = list(self.known_faces.keys())[match_index]
                    name = f"P{person_id}"  # Shorter name
                    confidence = 0.85  # Simplified confidence
                    self.update_face_sighting(person_id)
            
            if name == "Unknown" and len(self.known_faces) < 50:  # Limit database size
                face_crop = frame[max(0, top-20):min(frame.shape[0], bottom+20), 
                                max(0, left-20):min(frame.shape[1], right+20)]
                if face_crop.size > 0:
                    person_id = len(self.known_faces) + 1
                    self.add_new_face(face_encoding, face_crop, person_id)
                    name = f"P{person_id}"
            
            # Simple drawing (faster)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            face_info.append({
                'name': name,
                'location': (left, top, right, bottom)
            })
        
        return frame, face_info
    
    def initialize_lightweight_tracker(self, frame, bbox):
        """Initialize lightweight KCF tracker (faster than CSRT)"""
        try:
            self.tracker = cv2.TrackerKCF_create()
            success = self.tracker.init(frame, bbox)
            if success:
                self.tracking_active = True
                self.track_bbox = bbox
                self.stats['objects_tracked'] += 1
                logger.info("Lightweight tracking initialized")
            return success
        except:
            # Fallback to manual tracking if KCF not available
            self.tracker = None
            self.tracking_active = True
            self.track_bbox = bbox
            return True
    
    def update_tracker_lightweight(self, frame):
        """Update object tracker with fallback"""
        if not self.tracking_active:
            return frame, False
        
        if self.tracker:
            try:
                success, bbox = self.tracker.update(frame)
                if success:
                    self.track_bbox = bbox
                    x, y, w, h = [int(i) for i in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    cv2.putText(frame, "TRACK", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    return frame, True
                else:
                    self.tracking_active = False
                    return frame, False
            except:
                self.tracking_active = False
                return frame, False
        else:
            # Simple manual tracking fallback
            if self.track_bbox:
                x, y, w, h = [int(i) for i in self.track_bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 100), 1)
                cv2.putText(frame, "MANUAL", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            return frame, True
    
    def detect_motion_optimized(self, frame):
        """Optimized motion detection"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours with area filter
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 5000:  # Filter size range
                x, y, w, h = cv2.boundingRect(contour)
                motion_objects.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        
        return frame, motion_objects
    
    def add_minimal_hud(self, frame):
        """Minimal HUD overlay to reduce processing"""
        # Simple text overlay
        info_lines = [
            f"Faces: {len(self.known_faces)}",
            f"Track: {'ON' if self.tracking_active else 'OFF'}",
            f"FPS: {self.get_current_fps():.1f}"
        ]
        
        for i, text in enumerate(info_lines):
            cv2.putText(frame, text, (5, 15 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Simple crosshair
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (255, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (255, 255, 255), 1)
        
        return frame
    
    def get_current_fps(self):
        """Calculate current FPS efficiently"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                return (len(self.fps_counter) - 1) / time_diff
        return 0
    
    def save_snapshot_compressed(self, frame, prefix="snapshot"):
        """Save compressed snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.capture_path}{prefix}_{timestamp}.jpg"
        # High compression to save space
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        logger.info(f"Snapshot saved: {filename}")
        return filename
    
    def cleanup_memory(self):
        """Periodic memory cleanup"""
        if self.frame_count % 300 == 0:  # Every 300 frames
            gc.collect()
            
            # Limit face database size
            if len(self.known_faces) > 100:
                # Keep only most frequently seen faces
                sorted_faces = sorted(self.known_faces.items(), 
                                    key=lambda x: x[1].get('sightings', 0), reverse=True)
                self.known_faces = dict(sorted_faces[:50])
                logger.info("Face database trimmed to 50 entries")
    
    def run(self):
        """Optimized main execution loop"""
        logger.info("Starting Optimized Drone Vision System...")
        
        frame_time = 1.0 / self.target_fps
        last_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Frame rate control
                if current_time - last_time < frame_time:
                    time.sleep(0.001)  # Small sleep to prevent CPU spinning
                    continue
                
                # Capture frame
                try:
                    frame = self.picam2.capture_array()
                    if frame is None:
                        self.stats['frame_drops'] += 1
                        continue
                        
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"Frame capture failed: {e}")
                    continue
                
                self.frame_count += 1
                last_time = current_time
                
                # Process faces (reduced frequency)
                if self.frame_count % self.face_detection_interval == 0:
                    frame, face_info = self.process_faces_optimized(frame)
                
                # Update tracker (every frame for smoothness)
                if self.tracking_active:
                    frame, tracking_success = self.update_tracker_lightweight(frame)
                
                # Motion detection (reduced frequency)
                if self.frame_count % self.motion_detection_interval == 0:
                    frame, motion_objects = self.detect_motion_optimized(frame)
                
                # HUD update (reduced frequency)
                if self.frame_count % self.hud_update_interval == 0:
                    frame = self.add_minimal_hud(frame)
                
                # Scale up for display if needed
                if (self.frame_width != self.display_width or 
                    self.frame_height != self.display_height):
                    display_frame = cv2.resize(frame, (self.display_width, self.display_height))
                else:
                    display_frame = frame
                
                # Display
                cv2.imshow('Pi Drone Vision (Optimized)', display_frame)
                
                # Memory cleanup
                self.cleanup_memory()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_snapshot_compressed(frame)
                elif key == ord('t'):
                    # Start tracking first motion object if available
                    if 'motion_objects' in locals() and motion_objects:
                        bbox = motion_objects[0]
                        self.initialize_lightweight_tracker(frame, bbox)
                elif key == ord('r'):
                    self.tracking_active = False
                    self.tracker = None
                elif key == ord('c'):
                    self.known_faces.clear()
                    logger.info("Face database cleared")
                elif key == ord('p'):
                    # Performance info
                    logger.info(f"Frame drops: {self.stats['frame_drops']}, "
                              f"Memory faces: {len(self.known_faces)}")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        # Save face database
        self.save_face_database()
        
        # Save lightweight statistics
        self.stats['session_end'] = datetime.now()
        self.stats['total_frames'] = self.frame_count
        
        stats_file = f"{self.capture_path}stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(stats_file, 'w') as f:
                stats_copy = self.stats.copy()
                stats_copy['session_start'] = str(stats_copy['session_start'])
                stats_copy['session_end'] = str(stats_copy['session_end'])
                json.dump(stats_copy, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save stats: {e}")
        
        # Stop camera and cleanup
        try:
            self.picam2.stop()
        except:
            pass
        
        cv2.destroyAllWindows()
        
        logger.info("Cleanup complete")
        logger.info(f"Session: {len(self.known_faces)} faces, "
                   f"{self.stats['faces_detected']} detections, "
                   f"{self.frame_count} frames processed")

def main():
    """Main function with Pi optimization tips"""
    print("=== Pi-Optimized Drone Vision System ===")
    print("Resolution: 320x240 (scaled to 640x480 display)")
    print("Target FPS: 15 (optimized for Pi performance)")
    print("")
    print("Controls:")
    print("  q - Quit")
    print("  s - Save compressed snapshot")
    print("  t - Track motion object")
    print("  r - Reset tracking")
    print("  c - Clear face database")
    print("  p - Performance info")
    print("==========================================")
    print("")
    print("Pi Optimization Tips:")
    print("- Ensure Pi has adequate cooling")
    print("- Use fast SD card (Class 10+ or better)")
    print("- Close unnecessary programs")
    print("- Consider overclocking if stable")
    print("==========================================")
    
    try:
        vision_system = OptimizedDroneVisionSystem()
        vision_system.run()
    except Exception as e:
        logger.error(f"Failed to start vision system: {e}")
        print("\nTroubleshooting:")
        print("1. Check camera connection")
        print("2. Ensure libraries installed: pip install picamera2 opencv-python face-recognition numpy")
        print("3. Enable camera in raspi-config")
        print("4. Check Pi memory and CPU usage")

if __name__ == "__main__":
    main()
