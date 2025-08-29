# 🎬 Video Segment Extractor Using SIFT 🔍

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green?logo=opencv&logoColor=white)

Automatically detect a specific image in a video using **SIFT (Scale-Invariant Feature Transform)** and extract all frames containing that image into a **new video**!  

---

## 📜 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](./LICENSE) file for more details.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

---

## 🌟 Features

- 🔹 **Accurate Detection**: Finds the target image using SIFT keypoints.  
- 🔹 **Robust Matching**: Uses **Lowe’s ratio test** and **RANSAC** to reduce false positives.  
- 🔹 **Segment Extraction**: Saves matched frames as a **new video**.  
- 🔹 **Visualization**: Shows real-time matching using OpenCV.  
- 🔹 **Flexible**: Works with any video format supported by OpenCV & MoviePy.  

---

## 🛠️ Installation

```bash
pip install opencv-python numpy moviepy
```

---

## 📁 Project Structure
```
VideoSegmentExtractor/
│
├── extract_segments.py       # Main script with all functions
├── input/                    # Input videos & images
│   ├── sample_video.mp4
│   └── target_image.png
├── output/                   # Folder where matched segments are saved
├── docs/                     # Example screenshots / GIFs
└── README.md
```
---

## ⚡ How It Works

### 1. Image Preprocessing:
Converts the input image to grayscale and applies histogram equalization for better feature detection.

### 2. Feature Detection:
Detects SIFT keypoints and descriptors for both the input image and video frames.

### 3. Feature Matching:
Matches image features to each video frame using BFMatcher and applies Lowe’s ratio test to keep good matches.

### 4. Homography Filtering:
Uses RANSAC to filter outliers and keep only inlier matches for accurate detection.

### 5. Frame Extraction:
Frames with enough inlier matches are stored for output.

### 6. Video Creation:
Uses MoviePy to compile matched frames into a new video file.

---

## ▶️ Usage

input_video = "input/sample_video.mp4"
input_image = "input/target_image.png"
output_video = "output/matched_segments.mp4"


---
## 🖼️ Visualization
Displays matched keypoints in real-time for debugging:

---
## ⚙️ Configuration Options

```
| Parameter        | Default | Description                                                           |
| -----------------|---------|-----------------------------------------------------------------------|
| `ratio_threshold`| 0.75    | Lowe's ratio test threshold for feature matching                      |
| `min_inliers`    | 4       | Minimum number of inlier matches required to consider a frame matched |
| `fps`            | 30      | Frames per second for the output video                                |
```

💡 Tips:

- Adjust ratio_threshold in sift_match() for stricter or looser matching.
- Minimum 4 inlier matches are required to consider a frame as matched.


---

## 🔧 Advanced Usage

- Process large videos efficiently by skipping frames.
- Save individual frames as images instead of a video.
- Batch process multiple videos & images.

---

## 📌 Notes

- Use forward slashes / in paths for Windows compatibility.
- OpenCV windows can be disabled for headless processing.
- Works best on videos with minimal motion blur or occlusion.

---

<table>
<tr>
  <td>
    <ul>
      <li>🖼️ <strong>Input Image</strong> — The target image we are detecting in the video</li>
      <li>🎥 <strong>Input Video</strong> — The source video containing multiple frames</li>
      <li>✅ <strong>Output Video</strong> — Extracted frames where the image appears, compiled into a new video</li>
    </ul>
  </td>
  <td>
    <img src="Output\output_video.mp4" alt="Output Video GIF" width="300" height="300"/>
  </td>
</tr>
</table>

---


## 💡 Future Improvements

- ⚡ GPU acceleration for faster SIFT matching
- 🎯 Automatic scene detection
- 🔗 Support multiple input images
- 🖍 Annotated videos with bounding boxes for detected images

---

## 👩‍💻 Author
**Irene Betsy D** 

---
