import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import torch
from PIL import Image
import pathlib
import time
import pandas as pd
pathlib.PosixPath = pathlib.WindowsPath

def load_model():
    try:
        # Update these paths with your YOLOv5 directory and trained weights
        repo_path = "https://github.com/ultralytics/yolov5?tab=readme-ov-file"
        weights_path = "weight30/weights/best.pt"
        
        model = torch.hub.load(repo_path, "custom", path=weights_path, source="local", force_reload=True)
        st.success("YOLOv5 model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {str(e)}")
        return None

def detect_and_annotate(model, frame):
    try:
        results = model(frame)
        annotated_frame = frame.copy()
        detected_workers = {}
        
        for idx, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
            conf = float(conf)
            cls = int(cls)
            name = model.names[cls]
            
            if conf > 0.6 and name != 'unknown':  # Higher confidence and exclude unknown
                x1, y1, x2, y2 = map(int, xyxy)  # Convert coordinates to int
                
                # Store worker with their index
                detected_workers[idx] = {
                    'name': name,
                    'confidence': conf,
                }
                
                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, name, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame, detected_workers
    except Exception as e:
        st.error(f"Error in detection: {str(e)}")
        return frame, {}

def main():
    st.set_page_config(page_title="Real-Time Attendance System", layout="wide")
    st.title("Vision-Based Attendance System at Mahatma Gandhi NREGS")
    
    # Dropdown for Area and Group
    st.sidebar.header("Attendance Details")
    areas = ["Ponds Construction", "Rocks woks", "Water Conservation", "Road Maintenance", "Other"]
    groups = ["Group A", "Group B", "Group C", "Group D"]
    
    selected_area = st.sidebar.selectbox("Select Area", areas)
    selected_group = st.sidebar.selectbox("Select Group", groups)
    
    # Initialize session state
    if "attendance_records" not in st.session_state:
        st.session_state.attendance_records = []
    if "recorded_names" not in st.session_state:
        st.session_state.recorded_names = set()
    
    # Load YOLOv5 model
    model = load_model()
    if model is None:
        return
    
    # Camera selection
    st.sidebar.header("Camera Settings")
    available_cameras = ["Camera 0", "Camera 1"]
    selected_camera = st.sidebar.selectbox("Select Camera", available_cameras)
    camera_index = int(selected_camera[-1])
    
    # Webcam controls
    col1, col2, col3 = st.columns(3)
    with col1:
        start_webcam = st.button("Start Webcam")
    with col2:
        stop_webcam = st.button("Stop Webcam")
    with col3:
        # Timer display
        timer_placeholder = st.empty()
    
    # Frame display placeholder
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if start_webcam:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error(f"Failed to open camera {camera_index}. Trying alternative camera...")
            cap = cv2.VideoCapture(1 if camera_index == 0 else 0)
            
        if not cap.isOpened():
            st.error("No working camera found. Please check your camera connection.")
            return
        
        try:
            # Reset recorded names when starting webcam
            st.session_state.recorded_names.clear()
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.error("Failed to capture frame. Retrying...")
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
                if current_time - start_time <= 240:  # 4 minutes = 240 seconds
                    if detected_workers:
                        new_detections = []
                        
                        for idx, worker in detected_workers.items():
                            # Only record if name hasn't been recorded before
                            if worker['name'] not in st.session_state.recorded_names:
                                new_detections.append({
                                    'name': worker['name'],
                                    'confidence': worker['confidence'],
                                })
                                st.session_state.recorded_names.add(worker['name'])
                        
                        if new_detections:
                            record = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "area": selected_area,
                                "group": selected_group,
                                "workers": new_detections,
                            }
                            st.session_state.attendance_records.append(record)
                            
                            # Display names of newly recorded workers
                            detected_names = [det['name'] for det in new_detections]
                            status_placeholder.success(f"Attendance recorded for: {', '.join(detected_names)}")
                else:
                    timer_placeholder.success("Attendance Recording Complete!")
                    break
                
                if stop_webcam:
                    break
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
        except Exception as e:
            st.error(f"Error during webcam capture: {str(e)}")
        finally:
            cap.release()
            status_placeholder.success("Webcam stopped.")
    
    # Display attendance records
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
            
            # Get all unique classes from the worker detection
            unique_classes = list(known_workers)
            
            # Create a comprehensive attendance sheet
            for worker in unique_classes:
                present_records = [
                    record for record in st.session_state.attendance_records 
                    if worker in [w['name'] for w in record['workers']]
                ]
                
                # Determine attendance status
                present = len(present_records) > 0
                
                # Get the most recent record details if present
                if present_records:
                    latest_record = present_records[-1]
                    export_data.append({
                        'Name': worker,
                        'Area': latest_record['area'],
                        'Group': latest_record['group'],
                        'Attendance Status': 'Present' if present else 'Absent',
                        'Timestamp': latest_record['timestamp']
                    })
                else:
                    export_data.append({
                        'Name': worker,
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
                worker_names = [worker['name'] for worker in record['workers']]
                st.write(f"Workers Detected: {', '.join(worker_names)}")

if __name__ == "__main__":
    main()
# import streamlit as st
# import cv2
# from PIL import Image
# import numpy as np
# import os
# from datetime import datetime
# import torch
# import pathlib
# pathlib.PosixPath = pathlib.WindowsPath

# # Dummy user database (in production, use a proper database)
# users = {"admin@example.com": "admin123"}

# # Attendance records
# attendance = []
# import torch


# def load_model():
#     try:
#         repo_path = "/Users/balaji/Desktop/miniproject/yolov5"
#         weights_path = "/Users/balaji/Desktop/miniproject/yolov5/Experiments/weight_20/weights/best.pt"
#         from pathlib import Path
#         # Load the model using string paths
#         model = torch.hub.load(str(Path(repo_path)), "custom", path=str(Path(weights_path)), source="local", force_reload=True)  # local repo
#         st.success("YOLOv5 model loaded successfully!")
#         return model
#     except Exception as e:
#         st.error(f"Error loading YOLOv5 model: {str(e)}")
#         return None

# # YOLO Face Recognition Function
# def recognize_faces(image):

#     model = load_model()
#     if model is None:
#         return []
    
#     try:
#         # Convert PIL Image to numpy array if necessary
#         if isinstance(image, Image.Image):
#             img = np.array(image)
#         else:
#             img = image
        
#         # Perform detection
#         results = model(img)
        
#         # Extract detections with confidence scores
#         detections = []
#         for *xyxy, conf, cls in results.xyxy[0]:
#             conf = float(conf)
#             cls = int(cls)
            
#             # Get class name and filter by confidence
#             name = model.names[cls]
#             if conf > 0.5:  # Confidence threshold
#                 detections.append(name)
        
#         return detections
    
#     except Exception as e:
#         st.error(f"Error during detection: {str(e)}")
#         return []

# def main():
#     st.set_page_config(page_title="Attendance System", layout="wide")
#     st.title("Vision-Based Attendance System")

#     # Initialize session state
#     if "logged_in" not in st.session_state:
#         st.session_state.logged_in = False
#     if "attendance_records" not in st.session_state:
#         st.session_state.attendance_records = []

#     if not st.session_state.logged_in:
#         login_signup()
#     else:
#         attendance_page()

# def login_signup():
#     st.header("Login // Signup")
#     col1, col2 = st.columns(2)

#     with col1:
#         choice = st.radio("Choose an option", ["Login", "Signup"])

#     with col2:
#         if choice == "Signup":
#             with st.form("signup_form"):
#                 email = st.text_input("Email")
#                 password = st.text_input("Password", type="password")
#                 confirm_password = st.text_input("Confirm Password", type="password")
#                 submitted = st.form_submit_button("Signup")
                
#                 if submitted:
#                     if email in users:
#                         st.error("User already exists.")
#                     elif password != confirm_password:
#                         st.error("Passwords do not match.")
#                     else:
#                         users[email] = password
#                         st.success("Signup successful. Please login.")

#         if choice == "Login":
#             with st.form("login_form"):
#                 email = st.text_input("Email")
#                 password = st.text_input("Password", type="password")
#                 submitted = st.form_submit_button("Login")
                
#                 if submitted:
#                     if email in users and users[email] == password:
#                         st.session_state.logged_in = True
#                         st.session_state.email = email
#                         st.success("Login successful.")
#                         st.experimental_rerun()
#                     else:
#                         st.error("Invalid credentials.")

# def attendance_page():
#     st.sidebar.title("Options")
    
#     if st.sidebar.button("Logout"):
#         for key in list(st.session_state.keys()):
#             del st.session_state[key]
#         st.experimental_rerun()

#     st.header(f"Welcome, {st.session_state.email}")

#     # Create two columns for input and webcam capture
#     col1, col2 = st.columns(2)

#     with col1:
#         selected_date = st.date_input("Select Date", datetime.now())
#         selected_area = st.selectbox("Select Area", ["Area 1", "Area 2", "Area 3"])
#         group_name = st.selectbox("Select Group Name", ["Group A", "Group B", "Group C"])
        
#         uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        
#         if uploaded_image:
#             image = Image.open(uploaded_image)
#             st.image(image, caption="Uploaded Image", use_column_width=True)

#             if st.button("Process Image"):
#                 with st.spinner("Processing the image..."):
#                     names = recognize_faces(image)
                    
#                     if names:
#                         record = {
#                             "date": selected_date,
#                             "area": selected_area,
#                             "group": group_name,
#                             "workers": list(set(names)),
#                             "timestamp": datetime.now().strftime("%H:%M:%S"),
#                         }
#                         st.session_state.attendance_records.append(record)
#                         st.success(f"Recognized {len(set(names))} workers: {', '.join(set(names))}")
#                     else:
#                         st.warning("No workers detected in the image.")

#     with col2:
#         # Webcam Section
#         st.subheader("Webcam Capture")
        
#         if st.button("Start Webcam"):
#             try:
#                 cap = cv2.VideoCapture(0)
                
#                 if not cap.isOpened():
#                     st.error("Unable to access webcam. Please check your camera connection.")
#                     return
                
#                 stop_button = st.button("Stop Webcam")
#                 frame_placeholder = st.empty()
                
#                 while not stop_button:
#                     ret, frame = cap.read()
                    
#                     if not ret:
#                         st.error("Failed to capture frame from webcam.")
#                         break
                    
#                     names = recognize_faces(frame)

#                     # Draw detected names on frame
#                     for i, name in enumerate(names):
#                         cv2.putText(frame, name, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                     # Display frame
#                     frame_placeholder.image(frame, channels="BGR", use_column_width=True)

#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
                
#                 cap.release()

#                 if names:
#                     record = {
#                         "date": selected_date,
#                         "area": selected_area,
#                         "group": group_name,
#                         "workers": list(set(names)),
#                         "timestamp": datetime.now().strftime("%H:%M:%S"),
#                     }
#                     st.session_state.attendance_records.append(record)
#                     st.success(f"Attendance recorded for {len(set(names))} workers!")

#             except Exception as e:
#                 st.error(f"Error accessing webcam: {str(e)}")
#             finally:
#                 if 'cap' in locals():
#                     cap.release()

#     # Display Attendance Records
#     if len(st.session_state.attendance_records) > 0:
#         st.subheader("Attendance Records")
        
#         for record in reversed(st.session_state.attendance_records):
#             with st.expander(f"Record - {record['date']} - {record['timestamp']}"):
#                 st.write(f"Area: {record['area']}")
#                 st.write(f"Group: {record['group']}")
#                 st.write(f"Workers Present: {', '.join(record['workers'])}")

# if __name__ == "__main__":
#     main()

# # import streamlit as st
# # import cv2
# # from PIL import Image
# # import numpy as np
# # import os
# # from datetime import datetime
# # import torch

# # # Dummy user database (in production, use a proper database)
# # users = {"admin@example.com": "admin123"}

# # # Attendance records
# # attendance = []

# # def load_model():
# #     try:
# #         # Adjust the model path to where you've stored your trained weights
# #         model_path = r"C:/Users/balaji/Pictures/Vision_Based_Attendance_System_for_MGNREGA_Workers/YOLOV5/runs/train/exp/weights/best.pt"  # Use raw string for Windows paths
        
# #         # Check if the model file exists
# #         if not os.path.exists(model_path):
# #             st.error(f"Model file not found. Please check the path: {model_path}")
# #             return None
        
# #         # Load the model using torch (since it's a YOLOv5 model)
# #         model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)  # Added force_reload=True
# #         st.success("YOLOv5 model loaded successfully!")
# #         return model
# #     except Exception as e:
# #         st.error(f"Error loading YOLOv5 model: {str(e)}")
# #         return None



# # # YOLO Face Recognition Function
# # def recognize_faces(image):
# #     model = load_model()
# #     if model is None:
# #         return []
    
# #     try:
# #         # Convert PIL Image to numpy array if necessary
# #         if isinstance(image, Image.Image):
# #             img = np.array(image)
# #         else:
# #             img = image
        
# #         # Perform detection
# #         results = model(img)
        
# #         # Extract detections with confidence scores
# #         detections = []
# #         for *xyxy, conf, cls in results.xyxy[0]:
# #             conf = float(conf)
# #             cls = int(cls)
            
# #             # Get class name and filter by confidence
# #             name = model.names[cls]
# #             if conf > 0.5:  # Confidence threshold
# #                 detections.append(name)
        
# #         return detections
    
# #     except Exception as e:
# #         st.error(f"Error during detection: {str(e)}")
# #         return []

# # def main():
# #     st.set_page_config(page_title="Attendance System", layout="wide")
# #     st.title("Vision-Based Attendance System")

# #     # Initialize session state
# #     if "logged_in" not in st.session_state:
# #         st.session_state.logged_in = False
# #     if "attendance_records" not in st.session_state:
# #         st.session_state.attendance_records = []

# #     if not st.session_state.logged_in:
# #         login_signup()
# #     else:
# #         attendance_page()

# # def login_signup():
# #     st.header("Login // Signup")
# #     col1, col2 = st.columns(2)

# #     with col1:
# #         choice = st.radio("Choose an option", ["Login", "Signup"])

# #     with col2:
# #         if choice == "Signup":
# #             with st.form("signup_form"):
# #                 email = st.text_input("Email")
# #                 password = st.text_input("Password", type="password")
# #                 confirm_password = st.text_input("Confirm Password", type="password")
# #                 submitted = st.form_submit_button("Signup")
                
# #                 if submitted:
# #                     if email in users:
# #                         st.error("User already exists.")
# #                     elif password != confirm_password:
# #                         st.error("Passwords do not match.")
# #                     else:
# #                         users[email] = password
# #                         st.success("Signup successful. Please login.")

# #         if choice == "Login":
# #             with st.form("login_form"):
# #                 email = st.text_input("Email")
# #                 password = st.text_input("Password", type="password")
# #                 submitted = st.form_submit_button("Login")
                
# #                 if submitted:
# #                     if email in users and users[email] == password:
# #                         st.session_state.logged_in = True
# #                         st.session_state.email = email
# #                         st.success("Login successful.")
# #                         st.experimental_rerun()
# #                     else:
# #                         st.error("Invalid credentials.")

# # def attendance_page():
# #     st.sidebar.title("Options")
    
# #     if st.sidebar.button("Logout"):
# #         for key in list(st.session_state.keys()):
# #             del st.session_state[key]
# #         st.experimental_rerun()

# #     st.header(f"Welcome, {st.session_state.email}")

# #     # Create two columns for input and webcam capture
# #     col1, col2 = st.columns(2)

# #     with col1:
# #         selected_date = st.date_input("Select Date", datetime.now())
# #         selected_area = st.selectbox("Select Area", ["Area 1", "Area 2", "Area 3"])
# #         group_name = st.selectbox("Select Group Name", ["Group A", "Group B", "Group C"])
        
# #         uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        
# #         if uploaded_image:
# #             image = Image.open(uploaded_image)
# #             st.image(image, caption="Uploaded Image", use_column_width=True)

# #             if st.button("Process Image"):
# #                 with st.spinner("Processing the image..."):
# #                     names = recognize_faces(image)
                    
# #                     if names:
# #                         record = {
# #                             "date": selected_date,
# #                             "area": selected_area,
# #                             "group": group_name,
# #                             "workers": list(set(names)),
# #                             "timestamp": datetime.now().strftime("%H:%M:%S"),
# #                         }
# #                         st.session_state.attendance_records.append(record)
# #                         st.success(f"Recognized {len(set(names))} workers: {', '.join(set(names))}")
# #                     else:
# #                         st.warning("No workers detected in the image.")

# #     with col2:
# #         # Webcam Section
# #         st.subheader("Webcam Capture")
        
# #         if st.button("Start Webcam"):
# #             try:
# #                 cap = cv2.VideoCapture(0)
                
# #                 if not cap.isOpened():
# #                     st.error("Unable to access webcam. Please check your camera connection.")
# #                     return
                
# #                 stop_button = st.button("Stop Webcam")
# #                 frame_placeholder = st.empty()
                
# #                 while not stop_button:
# #                     ret, frame = cap.read()
                    
# #                     if not ret:
# #                         st.error("Failed to capture frame from webcam.")
# #                         break
                    
# #                     names = recognize_faces(frame)

# #                     # Draw detected names on frame
# #                     for i, name in enumerate(names):
# #                         cv2.putText(frame, name, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# #                     # Display frame
# #                     frame_placeholder.image(frame, channels="BGR", use_column_width=True)

# #                     if cv2.waitKey(1) & 0xFF == ord('q'):
# #                         break
                
# #                 cap.release()

# #                 if names:
# #                     record = {
# #                         "date": selected_date,
# #                         "area": selected_area,
# #                         "group": group_name,
# #                         "workers": list(set(names)),
# #                         "timestamp": datetime.now().strftime("%H:%M:%S"),
# #                     }
# #                     st.session_state.attendance_records.append(record)
# #                     st.success(f"Attendance recorded for {len(set(names))} workers!")

# #             except Exception as e:
# #                 st.error(f"Error accessing webcam: {str(e)}")
# #             finally:
# #                 if 'cap' in locals():
# #                     cap.release()

# #     # Display Attendance Records
# #     if len(st.session_state.attendance_records) > 0:
# #         st.subheader("Attendance Records")
        
# #         for record in reversed(st.session_state.attendance_records):
# #             with st.expander(f"Record - {record['date']} - {record['timestamp']}"):
# #                 st.write(f"Area: {record['area']}")
# #                 st.write(f"Group: {record['group']}")
# #                 st.write(f"Workers Present: {', '.join(record['workers'])}")

# # if __name__ == "__main__":
# #     main()
