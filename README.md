# 🔍 Multi Detection System

A real-time detection system using Python and OpenCV that detects:
- 😴 **Sleep/Drowsiness** — alerts when eyes are closed too long
- 🔥 **Fire** — alerts when fire color is detected in frame
- 😄 **Smile/Laugh** — alerts when a smile is detected

Each detection has its **own alarm sound!**

---

## 📸 How It Works

| Detection | How | Alarm |
|---|---|---|
| 😴 Sleep | Eye Aspect Ratio (EAR) < 0.25 for 20 frames | alarm_sleep.wav |
| 🔥 Fire | Orange/red color detection in frame | alarm_fire.wav |
| 😄 Smile | Mouth Aspect Ratio (MAR) > 0.6 for 10 frames | alarm_smile.wav |

---

## 💻 Requirements

- Python 3.10
- CMake
- Visual Studio Build Tools 2022 (Desktop development with C++)

---

## 📦 Installation

### 1. Clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/multi-detector.git
cd multi-detector
```

### 2. Install libraries
```bash
pip install opencv-python imutils scipy numpy pygame dlib
```

### 3. Download dlib model file
Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract and place `shape_predictor_68_face_landmarks.dat` in the project folder.

### 4. Add your sound files
Place your 3 alarm sounds in the `sounds/` folder:
```
sounds/
├── alarm_sleep.wav
├── alarm_fire.wav
└── alarm_smile.wav
```

---

## 🚀 Run

```bash
python detect.py
```

Press **Q** to quit.

---

## 📁 Project Structure

```
multi-detector/
├── detect.py                              ← main script
├── shape_predictor_68_face_landmarks.dat  ← download separately
├── sounds/
│   ├── alarm_sleep.wav                    ← add your own
│   ├── alarm_fire.wav                     ← add your own
│   └── alarm_smile.wav                    ← add your own
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes

- The `.dat` model file is **not included** (95MB — too large for GitHub)
- Sound files are **not included** — add your own `.wav` files
- Works best with **good lighting** and camera facing you directly
- Tested on **Windows 10/11** with Python 3.10

---

## 🛠️ Built With

- [OpenCV](https://opencv.org/)
- [dlib](http://dlib.net/)
- [imutils](https://github.com/jrosebr1/imutils)
- [pygame](https://www.pygame.org/)
- [scipy](https://scipy.org/)
