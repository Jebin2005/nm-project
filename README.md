# Hand Gesture Recognition from Video

This project detects hand gestures (number of fingers shown) from a video file using OpenCV and Python. It uses image processing techniques such as contour detection and convexity defects to identify gestures.

## 📸 How it Works

1. Loads a video (`input.mp4`) instead of webcam.
2. Detects hand regions using HSV color space.
3. Finds contours and convexity defects.
4. Counts fingers based on angles between defects.
5. Displays the detected gesture on each frame.


## 🔧 Requirements

- Python 3.8 or later  
- OpenCV  
- NumPy

### Install dependencies:
```bash
pip install opencv-python numpy## ✅ Features

- 🎥 Processes hand gestures from any video file  
- ✋ Detects finger count (1 to 5) using contour analysis  
- 📐 Uses convexity defect geometry to identify gaps between fingers  
- 💡 Visual feedback with labeled bounding boxes and contours  
- 🧠 Easily customizable for real-time webcam input or different gesture logic

---▶️ Usage
   1.Place your video in the project folder (e.g., input.mp4),    then  run:

   2.python Hand_Gesture.py

   3.To quit the video preview, press q.
   
## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgements

 -OpenCV contributors: https://opencv.org

 -HOG + SVM pedestrian detection method


## Feedback

If you have any feedback, please reach out to us at For questions or improvements, feel free to reach out:

📩 Email: Jebindon2005@gmail.com

🌐 GitHub: https://github.com/Jebin2005


