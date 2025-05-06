#!/usr/bin/env python3
"""
Test script for the SignDetector class.
"""

import cv2
import time
from sign_detector import SignDetector
from utils.config import Config
from utils.logger import setup_logger

def main():
    # Setup logging
    logger = setup_logger()
    logger.info("Starting SignDetector test")
    
    # Create configuration
    config = Config()
    
    # Initialize detector
    detector = SignDetector(config)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    logger.info("Starting test loop")
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Process frame with model
            processed_frame, letter, confidence = detector.process_frame(frame, visualize=True)
            
            # Display information
            cv2.putText(processed_frame, f"Detected: {letter} ({confidence:.2f})", 
                      (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Sign Detector Test', processed_frame)
            
            # Check for exit key
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Test completed")

if __name__ == "__main__":
    main()