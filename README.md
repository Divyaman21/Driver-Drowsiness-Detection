# Driver Drowsiness Detection System 🚗💤

A real-time driver drowsiness detection system using OpenCV, TensorFlow, and Pygame. This application monitors the driver's eyes via a webcam and triggers an alarm if signs of drowsiness (persistent eye closure) are detected.

## ✨ Features
- **Real-time Monitoring:** High-FPS face and eye tracking using Haar Cascades.
- **Deep Learning Inference:** Optimized TensorFlow model for classifying eye states (Open/Closed).
- **Modern HUD Overlay:** Dynamic UI showing FPS, Drowsiness level, and Alert status.
- **Audible Alarm:** Automatic alarm trigger when drowsiness threshold is exceeded.
- **GPU Acceleration:** Optimized for macOS Metal performance.

## 🛠️ Requirements
- Python 3.9+
- OpenCV
- TensorFlow 2.16+
- Pygame
- NumPy

## 🚀 Getting Started

1. **Install Dependencies:**
   ```bash
   pip install opencv-python tensorflow pygame numpy
   ```

2. **Run the Application:**
   ```bash
   python drowsiness_webcam.py
   ```

3. **Usage:**
   - Keep your face visible to the webcam.
   - The system will highlight eyes and track stability.
   - If you close your eyes for more than ~15 frames, the alarm will sound.
   - Press **'q'** to exit.

## 📁 Project Structure
- `drowsiness_webcam.py`: Main execution script.
- `drowsiness_model.h5`: Pre-trained Keras model for eye state detection.
- `alarm.wav`: Sound file for the alert system.
- `.gitignore`: Excludes unnecessary files from version control.

---
*Drive safely!* 🚗
