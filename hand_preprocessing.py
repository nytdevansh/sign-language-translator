#!/usr/bin/env python3
"""
Hand detection and preprocessing module for sign language recognition.
Uses MediaPipe for robust hand detection and landmark extraction.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging

class HandPreprocessor:
    """
    Handles hand detection, tracking, and preprocessing for sign language recognition.
    """
    
    def __init__(self, 
                 max_hands=1, 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 static_image_mode=False,
                 roi_padding=20,
                 target_size=(128, 128),
                 use_landmarks=True):
        """
        Initialize the hand preprocessor.
        
        Args:
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            static_image_mode: Whether to use static image mode (vs video)
            roi_padding: Padding around hand bounding box
            target_size: Target size for output images
            use_landmarks: Whether to use landmarks for additional features
        """
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Configuration
        self.roi_padding = roi_padding
        self.target_size = target_size
        self.use_landmarks = use_landmarks
        self.debug_mode = False
        
        self.logger.info("Hand preprocessor initialized")
        
    def set_debug_mode(self, debug=True):
        """Enable or disable debug mode for visualization."""
        self.debug_mode = debug
        
    def process_image(self, image):
        """
        Process an image to extract hand region and features.
        
        Args:
            image: Input BGR image
            
        Returns:
            dict: Processed results including:
                - 'hand_detected': Whether a hand was detected
                - 'processed_image': Processed image for model input
                - 'visualization': Debug visualization (if debug_mode is True)
                - 'landmarks': Hand landmarks (if use_landmarks is True)
                - 'hand_roi': Region of interest containing the hand
        """
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create result dictionary
        result = {
            'hand_detected': False,
            'processed_image': None,
            'visualization': None,
            'landmarks': None,
            'hand_roi': None
        }
        
        # Process with MediaPipe
        mp_results = self.hands.process(rgb_image)
        
        # Create visualization copy if in debug mode
        if self.debug_mode:
            vis_image = image.copy()
        
        # Check if hand landmarks are detected
        if mp_results.multi_hand_landmarks:
            result['hand_detected'] = True
            
            # Get the first detected hand
            hand_landmarks = mp_results.multi_hand_landmarks[0]
            
            # Draw landmarks if in debug mode
            if self.debug_mode:
                self.mp_drawing.draw_landmarks(
                    vis_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Extract hand region from the image
            hand_roi, roi_coords = self._extract_hand_region(image, hand_landmarks)
            
            if hand_roi is not None:
                # Store ROI
                result['hand_roi'] = hand_roi
                
                # Preprocess for model input
                processed = cv2.resize(hand_roi, self.target_size)
                
                # Normalize
                processed = processed / 255.0
                
                # Store processed image
                result['processed_image'] = processed
                
                # Draw ROI on visualization
                if self.debug_mode:
                    x1, y1, x2, y2 = roi_coords
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extract and store landmarks if needed
            if self.use_landmarks:
                result['landmarks'] = self._extract_landmarks(hand_landmarks, image.shape)
        
        # Store visualization if in debug mode
        if self.debug_mode:
            result['visualization'] = vis_image
            
        return result
    
    def _extract_hand_region(self, image, landmarks):
        """
        Extract the hand region from the image using landmarks.
        
        Args:
            image: Input BGR image
            landmarks: MediaPipe hand landmarks
            
        Returns:
            tuple: (hand_roi, roi_coordinates)
        """
        h, w, _ = image.shape
        
        # Initialize bounding box coordinates
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        # Find bounding box around hand landmarks
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Add padding
        x_min = max(0, x_min - self.roi_padding)
        y_min = max(0, y_min - self.roi_padding)
        x_max = min(w, x_max + self.roi_padding)
        y_max = min(h, y_max + self.roi_padding)
        
        # Make square to avoid distortion
        width = x_max - x_min
        height = y_max - y_min
        if width > height:
            # Add to height
            diff = width - height
            y_min = max(0, y_min - diff // 2)
            y_max = min(h, y_max + diff // 2)
        else:
            # Add to width
            diff = height - width
            x_min = max(0, x_min - diff // 2)
            x_max = min(w, x_max + diff // 2)
        
        # Extract region
        hand_roi = image[y_min:y_max, x_min:x_max]
        
        # Return None if ROI is empty
        if hand_roi.size == 0:
            return None, (x_min, y_min, x_max, y_max)
        
        return hand_roi, (x_min, y_min, x_max, y_max)
    
    def _extract_landmarks(self, landmarks, image_shape):
        """
        Extract normalized landmarks from MediaPipe results.
        
        Args:
            landmarks: MediaPipe hand landmarks
            image_shape: Shape of the original image
            
        Returns:
            np.ndarray: Array of landmark coordinates
        """
        h, w, _ = image_shape
        landmark_array = []
        
        # Convert landmarks to array format
        for landmark in landmarks.landmark:
            # Normalize coordinates to 0-1
            landmark_array.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmark_array)
    
    def release(self):
        """Release resources."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()

# Example usage
if __name__ == "__main__":
    # Example script to test the preprocessor
    import matplotlib.pyplot as plt
    
    # Initialize preprocessor
    preprocessor = HandPreprocessor(use_landmarks=True)
    preprocessor.set_debug_mode(True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            result = preprocessor.process_image(frame)
            
            # Display results
            if result['visualization'] is not None:
                cv2.imshow('Hand Detection', result['visualization'])
                
                if result['hand_roi'] is not None:
                    cv2.imshow('Hand ROI', result['hand_roi'])
                    
                    if result['processed_image'] is not None:
                        # Convert to displayable format
                        display_img = (result['processed_image'] * 255).astype(np.uint8)
                        cv2.imshow('Processed Image', display_img)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        preprocessor.release()