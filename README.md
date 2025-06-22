# Vision-Based Attendance System for MGNREGA Workers

A computer vision-based attendance system designed specifically for Mahatma Gandhi National Rural Employment Guarantee Act (MGNREGA) workers. This system uses YOLOv5 for real-time face detection and recognition to automate attendance tracking.

## 🎯 Project Overview

This system provides an automated solution for tracking attendance of MGNREGA workers using computer vision technology. It can recognize workers in real-time through webcam feeds or process uploaded images to record attendance automatically.

## ✨ Features

- **Real-time Face Recognition**: Uses YOLOv5 model for accurate worker identification
- **Multiple Detection Modes**:
  - Webcam mode for live attendance recording
  - Single image upload for individual processing
  - Bulk image upload for batch processing
- **Text-to-Speech Announcements**: Audio confirmation when workers are detected
- **Attendance Management**: 
  - Automatic attendance recording with timestamps
  - Area and group-based organization
  - Export functionality to Excel format
- **Worker Database**: Pre-configured database with unique IDs for each worker
- **User-friendly Interface**: Streamlit-based web application

## 👥 Supported Workers

The system is trained to recognize the following workers:
- BALAJI (ID: R190014)
- CHAND (ID: R190024)
- NARAYANA (ID: R190027)
- PAVAN (ID: R19111)
- RAJU (ID: R190551)
- SANTHOSH (ID: R19112)
- SUNNY (ID: R19117)

## 🏗️ Project Structure

```
miniproject/
├── README.md                           # This file
├── yolov5/                             # YOLOv5 framework
│   ├── Experiments/
│   │   ├── attendance_app.py           # Main Streamlit application
│   │   ├── datasetk/                   # Training dataset
│   │   │   ├── data.yaml              # Dataset configuration
│   │   │   ├── train/                 # Training images
│   │   │   ├── valid/                 # Validation images
│   │   │   └── test/                  # Test images
│   │   └── runs/                      # Training outputs
│   └── [YOLOv5 framework files]
├── test.py                            # Model testing script
├── venv/                              # Python virtual environment
└── [Presentation files]
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Webcam (for live detection mode)
- Windows/Linux/macOS

### Step 1: Clone the Repository

```bash
git clone https://github.com/KimavathBalajiNayak210/Vision-Based-Attendance-System-at-Mahatma-Gandhi-NREGS.git
cd Vision-Based-Attendance-System-at-Mahatma-Gandhi-NREGS
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install torch torchvision
pip install streamlit opencv-python pillow pandas pyttsx3 openpyxl
```

### Step 4: Download YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
```

## 🎮 Usage

### Running the Application

1. **Activate the virtual environment**:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run yolov5/Experiments/attendance_app.py
   ```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

### Using the Application

1. **Select Area and Group**: Choose the work area and group from the sidebar
2. **Choose Detection Mode**:
   - **Webcam**: Real-time attendance recording (4-minute session)
   - **Image Upload**: Process single images
   - **Bulk Image Upload**: Process multiple images at once
3. **Record Attendance**: The system will automatically detect and record worker attendance
4. **Export Data**: Download attendance records as Excel files

### Testing the Model

To test the YOLOv5 model independently:

```bash
python test.py
```

## 📊 Dataset Information

- **Source**: Kuruvupani dataset from Roboflow
- **Total Images**: 4,417 images
- **Classes**: 7 workers
- **Format**: YOLO v5 PyTorch format
- **Preprocessing**: Auto-orientation and resize to 640x640

## 🔧 Configuration

### Model Weights
The system uses a custom-trained YOLOv5 model. The weights are located at:
```
yolov5/Experiments/runs/train/exp/weights/best.pt
```

### Worker Database
Worker information is stored in the `WORKER_DATABASE` dictionary in `attendance_app.py`. You can modify this to add or remove workers.

## 📈 Features in Detail

### Real-time Detection
- 4-minute webcam sessions for attendance recording
- Automatic worker identification with confidence scores
- Visual bounding boxes and labels
- Audio announcements for detected workers

### Attendance Management
- Timestamp-based recording
- Area and group categorization
- Duplicate detection prevention
- Session-based tracking

### Export Functionality
- Excel format export
- Comprehensive attendance sheets
- Sorted by attendance status
- Includes worker IDs and timestamps

## 🛠️ Technical Details

- **Framework**: YOLOv5 (PyTorch)
- **Web Framework**: Streamlit
- **Computer Vision**: OpenCV
- **Text-to-Speech**: pyttsx3
- **Data Processing**: Pandas
- **Export**: OpenPyXL

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kimavath Balaji Nayak**
- GitHub: [@KimavathBalajiNayak210](https://github.com/KimavathBalajiNayak210)

## 🙏 Acknowledgments

- YOLOv5 by Ultralytics
- Roboflow for dataset hosting
- Streamlit for the web framework
- MGNREGA for the use case inspiration

## 📞 Support

For support and questions, please open an issue on the GitHub repository or contact the author.

---

**Note**: This system is specifically designed for MGNREGA workers and may need customization for other use cases. The model is trained on a specific dataset and may require retraining for different worker groups.
