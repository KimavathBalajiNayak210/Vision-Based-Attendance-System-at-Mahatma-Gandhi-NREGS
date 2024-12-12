

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import torch
from PIL import Image
import pathlib
import time
import pandas as pd
import pyttsx3

# Ensure Windows path compatibility
pathlib.PosixPath = pathlib.WindowsPath

# Predefined worker database with unique IDs
WORKER_DATABASE = {
    'BALAJI': 'R190014',
    'CHAND': 'R190024',
    'NARAYANA': 'R190027',
    'PAVAN': 'R19111',
    'RAJU': 'R190551',
    'SANTHOSH': 'R19112',
    'SUNNY': 'R19117'
}

# Initialize text-to-speech engine
def initialize_tts():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        return engine
    except Exception as e:
        st.error(f"Text-to-Speech initialization error: {str(e)}")
        return None

# Speak worker name function
def speak_worker_name(tts_engine, worker_name, unique_id):
    if tts_engine:
        try:
            announcement = f"Attendance recorded for {worker_name}, Employee ID {unique_id}"
            tts_engine.say(announcement)
            tts_engine.runAndWait()
        except Exception as e:
            st.error(f"Speech announcement error: {str(e)}")

# Load YOLOv5 model
def load_model():
    try:
        # Update these paths with your YOLOv5 directory and trained weights
        repo_path = "/Users/balaji/Desktop/miniproject/yolov5"
        weights_path = "/Users/balaji/Desktop/miniproject/yolov5/Experiments/weightLarge36/ml36/weights/best.pt"
        
        model = torch.hub.load(repo_path, "custom", path=weights_path, source="local", force_reload=True)
        st.success("YOLOv5 model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {str(e)}")
        return None

# Detect and annotate workers
def detect_and_annotate(model, frame):
    try:
        results = model(frame)
        annotated_frame = frame.copy()
        detected_workers = {}
        
        for idx, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
            conf = float(conf)
            cls = int(cls)
            name = model.names[cls]
            
            if conf > 0.7 and name != 'unknown':  # Higher confidence and exclude unknown
                x1, y1, x2, y2 = map(int, xyxy)  # Convert coordinates to int
           
                # Store worker with their index and unique ID
                detected_workers[idx] = {
                    'name': name,
                    'confidence': conf,
                    'unique_id': WORKER_DATABASE.get(name, 'N/A')
                }
                
                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} ({WORKER_DATABASE.get(name, 'N/A')})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame, detected_workers
    except Exception as e:
        st.error(f"Error in detection: {str(e)}")
        return frame, {}

# Process multiple uploaded files
def process_uploaded_files(model, tts_engine, selected_area, selected_group):
    uploaded_files = st.file_uploader(
        "Upload Attendance Images", 
        type=['jpg', 'jpeg', 'png', 'bmp'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Read the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Detect workers in the uploaded image
                annotated_image, detected_workers = detect_and_annotate(model, image)
                
                # Display the annotated image
                st.image(annotated_image, channels="BGR", caption=uploaded_file.name)
                
                # Record attendance for detected workers
                if detected_workers:
                    new_detections = []
                    
                    for idx, worker in detected_workers.items():
                        # Only record if name hasn't been recorded before
                        if worker['name'] not in st.session_state.recorded_workers:
                            new_detections.append({
                                'name': worker['name'],
                                'confidence': worker['confidence'],
                                'unique_id': worker['unique_id']
                            })
                            
                            # Automatically speak worker name when detected
                            if tts_engine:
                                speak_worker_name(tts_engine, worker['name'], worker['unique_id'])
                            
                            st.session_state.recorded_workers.add(worker['name'])
                    
                    if new_detections:
                        record = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "area": selected_area,
                            "group": selected_group,
                            "workers": new_detections,
                            "source_file": uploaded_file.name
                        }
                        st.session_state.attendance_records.append(record)
                        
                        # Display names of newly recorded workers
                        detected_names = [f"{det['name']} ({det['unique_id']})" for det in new_detections]
                        st.success(f"Attendance recorded for: {', '.join(detected_names)}")
                else:
                    st.warning(f"No workers detected in {uploaded_file.name}")
            
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Main application
def main():
    st.set_page_config(page_title="Attendance System", layout="wide")
    st.title("Vision-Based Attendance System at MGNREGA")
    
    # Initialize Text-to-Speech
    tts_engine = initialize_tts()
    
    # Dropdown for Area and Group
    st.sidebar.header("Attendance Details")
    areas = ["Ponds Construction", "Rocks Works", "Water Conservation", "Road Maintenance", "Other"]
    groups = ["Group A", "Group B", "Group C", "Group D"]
    
    selected_area = st.sidebar.selectbox("Select Area", areas)
    selected_group = st.sidebar.selectbox("Select Group", groups)
    
    # Initialize session state
    if "attendance_records" not in st.session_state:
        st.session_state.attendance_records = []
    if "recorded_workers" not in st.session_state:
        st.session_state.recorded_workers = set()
    
    # Load YOLOv5 model
    model = load_model()
    if model is None:
        return
    
    # Mode selection
    mode = st.sidebar.radio("Select Detection Mode", ["Webcam", "Image Upload", "Bulk Image Upload"])
    
    # Frame display placeholder
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Webcam Mode
    if mode == "Webcam":
        # Webcam controls
        col1, col2, col3 = st.columns(3)
        with col1:
            start_webcam = st.button("Start Webcam")
        with col2:
            stop_webcam = st.button("Stop Webcam")
        with col3:
            # Timer display
            timer_placeholder = st.empty()
        
        if start_webcam:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Failed to open camera. Trying alternative camera...")
                cap = cv2.VideoCapture(1)
                
            if not cap.isOpened():
                st.error("No working camera found.")
                return
            
            try:
                # Reset recorded workers when starting webcam
                st.session_state.recorded_workers.clear()
                start_time = time.time()
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.error("Failed to capture frame.")
                        time.sleep(1)
                        continue
                    
                    # Calculate remaining time
                    current_time = time.time()
                    remaining_time = max(0, 240 - (current_time - start_time))
                    
                    # Display timer
                    timer_placeholder.warning(f"Time Remaining: {int(remaining_time)} seconds")
                    
                    # Detect and annotate faces
                    annotated_frame, detected_workers = detect_and_annotate(model, frame)
                    
                    # Display frame
                    frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                    
                    # Record attendance within 4 minutes
                    if current_time - start_time <= 240:
                        if detected_workers:
                            new_detections = []
                            
                            for idx, worker in detected_workers.items():
                                # Only record if name hasn't been recorded before
                                if worker['name'] not in st.session_state.recorded_workers:
                                    new_detections.append({
                                        'name': worker['name'],
                                        'confidence': worker['confidence'],
                                        'unique_id': worker['unique_id']
                                    })
                                    
                                    # Automatically speak worker name when detected
                                    if tts_engine:
                                        speak_worker_name(tts_engine, worker['name'], worker['unique_id'])
                                    
                                    st.session_state.recorded_workers.add(worker['name'])
                            
                            if new_detections:
                                record = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "area": selected_area,
                                    "group": selected_group,
                                    "workers": new_detections,
                                }
                                st.session_state.attendance_records.append(record)
                                
                                # Display names of newly recorded workers
                                detected_names = [f"{det['name']} ({det['unique_id']})" for det in new_detections]
                                status_placeholder.success(f"Attendance recorded for: {', '.join(detected_names)}")
                    else:
                        timer_placeholder.success("Attendance Recording Complete!")
                        break
                    
                    if stop_webcam:
                        break
                    
                    time.sleep(0)
                
            except Exception as e:
                st.error(f"Error during webcam capture: {str(e)}")
            finally:
                cap.release()
                status_placeholder.success("Webcam stopped.")
    
    # Single Image Upload Mode
    elif mode == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            try:
                # Read the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Detect workers in the uploaded image
                annotated_image, detected_workers = detect_and_annotate(model, image)
                
                # Display the annotated image
                frame_placeholder.image(annotated_image, channels="BGR", use_column_width=True)
                
                # Record attendance for detected workers
                if detected_workers:
                    new_detections = []
                    
                    for idx, worker in detected_workers.items():
                        # Only record if name hasn't been recorded before
                        if worker['name'] not in st.session_state.recorded_workers:
                            new_detections.append({
                                'name': worker['name'],
                                'confidence': worker['confidence'],
                                'unique_id': worker['unique_id']
                            })
                            
                            # Automatically speak worker name when detected
                            if tts_engine:
                                speak_worker_name(tts_engine, worker['name'], worker['unique_id'])
                            
                            st.session_state.recorded_workers.add(worker['name'])
                    
                    if new_detections:
                        record = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "area": selected_area,
                            "group": selected_group,
                            "workers": new_detections,
                        }
                        st.session_state.attendance_records.append(record)
                        
                        # Display names of newly recorded workers
                        detected_names = [f"{det['name']} ({det['unique_id']})" for det in new_detections]
                        status_placeholder.success(f"Attendance recorded for: {', '.join(detected_names)}")
                else:
                    status_placeholder.warning("No workers detected in the uploaded image.")
            
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
    
    # Bulk Image Upload Mode
    elif mode == "Bulk Image Upload":
        process_uploaded_files(model, tts_engine, selected_area, selected_group)
    
    # Display and Export Attendance Records
    if st.session_state.attendance_records:
        st.subheader("Attendance Records")
        
        # Prepare attendance sheet for export
        all_workers = set()
        for record in st.session_state.attendance_records:
            all_workers.update([worker['name'] for worker in record['workers']])
        
        # Add export button
        if st.button("Export Attendance Records"):
            # Prepare data for Excel export
            export_data = []
            
            # All known workers from the model
            known_workers = list(all_workers)
            
            # Create a comprehensive attendance sheet
            for worker in known_workers:
                present_records = [
                    record for record in st.session_state.attendance_records 
                    if worker in [w['name'] for w in record['workers']]
                ]
                
                # Determine attendance status
                present = len(present_records) > 0
                
                # Get the most recent record details if present
                if present_records:
                    latest_record = present_records[-1]
                    unique_id = next((w['unique_id'] for w in latest_record['workers'] if w['name'] == worker), 'N/A')
                    export_data.append({
                        'Name': worker,
                        'Unique ID': unique_id,
                        'Area': latest_record['area'],
                        'Group': latest_record['group'],
                        'Attendance Status': 'Present' if present else 'Absent',
                        'Timestamp': latest_record['timestamp']
                    })
                else:
                    export_data.append({
                        'Name': worker,
                        'Unique ID': WORKER_DATABASE.get(worker, 'N/A'),
                        'Area': selected_area,
                        'Group': selected_group,
                        'Attendance Status': 'Absent',
                        'Timestamp': 'N/A'
                    })
            
            # Create DataFrame
            df = pd.DataFrame(export_data)
            
            # Sort DataFrame by Attendance Status (Present first)
            df = df.sort_values('Attendance Status', ascending=False)
            
            # Create Excel file
            excel_file = f"attendance_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            # Use BytesIO to create an in-memory file
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Attendance')
            
            # Download button
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=excel_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Display records
        for record in reversed(st.session_state.attendance_records):
            with st.expander(f"Timestamp: {record['timestamp']} | Area: {record['area']} | Group: {record['group']}"):
                worker_details = [f"{worker['name']} (ID: {worker['unique_id']})" for worker in record['workers']]
                st.write(f"Workers Detected: {', '.join(worker_details)}")

if __name__ == "__main__":
    main()


