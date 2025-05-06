# Sign Language Translator

A real-time sign language translation application using computer vision and deep learning.

## Features

- Real-time sign language detection and translation
- Support for American Sign Language (ASL) alphabet
- Multiple operation modes:
  - Inference: Translate signs in real-time
  - Training: Train the model with your own data
  - Data Collection: Collect training data for custom signs
- Configurable settings for different environments and use cases

## Requirements

- Python 3.7+
- TensorFlow 2.5+
- OpenCV 4.5+
- MediaPipe 0.8.10+
- NumPy 1.19+

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirement.txt
   ```
3. Run the application:
   ```
   python main.py
   ```

## Usage

### Basic Usage

Run the application in inference mode (default):

```
python main.py
```

### Command Line Arguments

- `--model`: Path to the trained model (default: models/sign_language_model.h5)
- `--camera`: Camera index to use (default: 0)
- `--mode`: Operation mode (inference, training, data_collection)
- `--dataset`: Dataset directory for training or collection
- `--debug`: Enable debug mode with visualization
- `--subtitle_size`: Size multiplier for subtitles
- `--word_timeout`: Timeout in seconds to form words

### Examples

Collect training data:
```
python main.py --mode data_collection --dataset my_dataset
```

Train the model:
```
python main.py --mode training --dataset my_dataset --model my_model.h5
```

Run inference with debug visualization:
```
python main.py --debug
```

## Controls

- Press `ESC` to exit the application
- Press `c` to clear the current text

## Project Structure

- `main.py`: Main application entry point
- `sign_detector.py`: Core sign detection functionality
- `utils/`: Utility modules
  - `config.py`: Configuration management
  - `logger.py`: Logging setup
  - `landmark_processor.py`: Hand landmark processing
- `models/`: Directory for trained models
- `dataset/`: Default directory for training data

## License

This project is licensed under the MIT License - see the LICENSE file for details.
