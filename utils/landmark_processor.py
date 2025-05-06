#!/usr/bin/env python3
"""
Process hand landmarks from MediaPipe for more efficient feature extraction.
"""

import numpy as np
import logging

class LandmarkProcessor:
    def __init__(self):
        """Initialize the landmark processor"""
        self.logger = logging.getLogger(__name__)
    
    def extract_landmarks(self, hand_landmarks):
        """
        Extract raw landmarks from MediaPipe hand_landmarks object
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            Array of landmark coordinates (x, y, z)
        """
        landmarks_array = []
        for landmark in hand_landmarks.landmark:
            landmarks_array.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks_array)
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks to make them invariant to translation and scale
        
        Args:
            landmarks: Raw landmarks array
            
        Returns:
            Normalized landmarks
        """
        # Reshape to (21, 3) for easier processing
        landmarks = landmarks.reshape(-1, 3)
        
        # Get wrist position (landmark 0)
        wrist = landmarks[0]
        
        # Translate all points relative to wrist
        centered = landmarks - wrist
        
        # Find the distance between wrist and middle finger MCP (landmark 9)
        # to use as normalization reference
        scale_reference = np.linalg.norm(centered[9])
        if scale_reference > 0:
            normalized = centered / scale_reference
        else:
            normalized = centered
            self.logger.warning("Scale reference is zero, skipping scale normalization")
        
        # Flatten back to 1D array
        return normalized.flatten()
    
    def extract_angles(self, landmarks):
        """
        Extract angles between finger joints for better pose representation
        
        Args:
            landmarks: Raw or normalized landmarks array
            
        Returns:
            Array of joint angles
        """
        # Reshape to (21, 3) for easier processing
        landmarks = landmarks.reshape(-1, 3)
        
        # Finger joint connections (each sublist represents one finger)
        finger_connections = [
            [0, 1, 2, 3, 4],       # Thumb
            [0, 5, 6, 7, 8],       # Index
            [0, 9, 10, 11, 12],    # Middle
            [0, 13, 14, 15, 16],   # Ring
            [0, 17, 18, 19, 20],   # Pinky
        ]
        
        angles = []
        for finger in finger_connections:
            for i in range(1, len(finger) - 1):
                # Get three consecutive joints
                p1 = landmarks[finger[i-1]]
                p2 = landmarks[finger[i]]
                p3 = landmarks[finger[i+1]]
                
                # Compute vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Compute angle (in radians)
                try:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    # Clip to handle floating point errors
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
                except:
                    # Handle zero-length vectors
                    angles.append(0)
        
        return np.array(angles)
    
    def extract_features(self, hand_landmarks):
        """
        Extract comprehensive feature set from hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            Feature vector combining different feature types
        """
        # Extract raw landmarks
        raw_landmarks = self.extract_landmarks(hand_landmarks)
        
        # Normalize landmarks
        normalized_landmarks = self.normalize_landmarks(raw_landmarks)
        
        # Extract angles
        angles = self.extract_angles(normalized_landmarks)
        
        # Combine features (you can adjust which features to include)
        features = np.concatenate([normalized_landmarks, angles])
        
        return features