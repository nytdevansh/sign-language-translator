#!/usr/bin/env python3
"""
Configuration management for the Sign Language Translator application.
"""

import os
import json
import logging

class Config:
    def __init__(self):
        """Initialize configuration with default values"""
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.model_path = 'models/sign_language_model.h5'
        self.mode = 'inference'
        self.debug = False
        self.camera_index = 0
        self.dataset_dir = 'dataset'
        self.subtitle_size = 1.0
        self.word_timeout = 2.0
        
        # Advanced model parameters
        self.input_size = (128, 128)
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 30
        
        # Processing parameters
        self.confidence_threshold = 0.7
        self.stable_duration = 0.5
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def load_from_file(self, config_file):
        """
        Load configuration from a JSON file
        
        Args:
            config_file: Path to JSON configuration file
        """
        if not os.path.exists(config_file):
            self.logger.warning(f"Config file {config_file} not found, using defaults")
            return
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update config with values from file
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    self.logger.debug(f"Set config {key} = {value}")
                else:
                    self.logger.warning(f"Unknown config parameter: {key}")
            
            self.logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
    
    def load_from_args(self, args):
        """
        Load configuration from command line arguments
        
        Args:
            args: Parsed command line arguments
        """
        if hasattr(args, 'model') and args.model:
            self.model_path = args.model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        if hasattr(args, 'mode') and args.mode:
            self.mode = args.mode
        
        if hasattr(args, 'debug'):
            self.debug = args.debug
        
        if hasattr(args, 'camera') and args.camera is not None:
            self.camera_index = args.camera
        
        if hasattr(args, 'dataset') and args.dataset:
            self.dataset_dir = args.dataset
        
        if hasattr(args, 'subtitle_size') and args.subtitle_size is not None:
            self.subtitle_size = args.subtitle_size
        
        if hasattr(args, 'word_timeout') and args.word_timeout is not None:
            self.word_timeout = args.word_timeout
        
        self.logger.info(f"Configuration loaded from command line arguments")
    
    def save_to_file(self, config_file):
        """
        Save current configuration to a JSON file
        
        Args:
            config_file: Path to save configuration
        """
        # Create config directory if needed
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        try:
            # Convert config to dictionary
            config_dict = {
                "model_path": self.model_path,
                "mode": self.mode,
                "debug": self.debug,
                "camera_index": self.camera_index,
                "dataset_dir": self.dataset_dir,
                "subtitle_size": self.subtitle_size,
                "word_timeout": self.word_timeout,
                "input_size": self.input_size,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "confidence_threshold": self.confidence_threshold,
                "stable_duration": self.stable_duration
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
            
            self.logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Error saving config file: {e}")
    
    def display(self):
        """Display current configuration settings"""
        self.logger.info("Current configuration:")
        self.logger.info(f"  Model path: {self.model_path}")
        self.logger.info(f"  Mode: {self.mode}")
        self.logger.info(f"  Debug: {self.debug}")
        self.logger.info(f"  Camera index: {self.camera_index}")
        self.logger.info(f"  Dataset directory: {self.dataset_dir}")
        self.logger.info(f"  Subtitle size: {self.subtitle_size}")
        self.logger.info(f"  Word timeout: {self.word_timeout}")
        self.logger.info(f"  Input size: {self.input_size}")
        self.logger.info(f"  Batch size: {self.batch_size}")
        self.logger.info(f"  Learning rate: {self.learning_rate}")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"  Stable duration: {self.stable_duration}")