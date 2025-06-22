# AI Attendance System using Face Recognition (OpenCV and dlib)

![Attendance System](https://img.shields.io/badge/status-active-brightgreen)

## Overview

This project is a **real-time face recognition attendance system** that uses a webcam to detect and recognize faces from a known dataset and automatically logs attendance in a CSV file. It uses the Python library `face_recognition`, which relies on `dlib` for face detection and face encoding.

---

## Features

- Load and encode known faces from images in the `known_faces` folder
- Detect and recognize faces from webcam video stream
- Mark attendance with timestamp in `attendance.csv`
- Display bounding boxes and names on recognized faces in real-time

---

## Prerequisites

- Python 3.8 or higher  
- Required Python packages:
  - `face_recognition`
  - `opencv-python`
  - `numpy`

> **Note:** Installing `dlib` (a dependency of `face_recognition`) can be challenging on Windows and may require Visual Studio Build Tools and CMake. Refer to official documentation if you encounter issues.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/ai-attendance-system.git
   cd ai-attendance-system
