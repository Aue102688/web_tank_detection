import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import subprocess
import datetime
import torch
import glob
import sys
import cv2
import os

# Custom CSS for input styles
st.markdown(
    '''
    <style>
        .stTextInput > div > div > input { color: white; }
        .stTextInput > div > div > input::placeholder { color: white; }
    </style>
    ''', 
    unsafe_allow_html=True
)

# Logo and Header
st.markdown("<div style='text-align: center;'><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/7-eleven_logo.svg/791px-7-eleven_logo.svg.png' width='120'></div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Water Preventive Maintenance Classification 💦</h1>", unsafe_allow_html=True)

# Sidebar for settings
st.sidebar.title("Settings")
st.sidebar.write("Adjust settings as needed.")

# Path of IMAGE & CSV
documents_path_img = os.path.join(os.path.expanduser("~"), "Documents", "YOLOAppData", "IMAGE_file")
documents_path_csv = os.path.join(os.path.expanduser("~"), "Documents", "YOLOAppData", "CSV_file")

# Func - For pull csv 
# (Load many csv-file)
def load_csv_data():
    # var folder_path of csv for go to All csv file 
    folder_path = os.path.join(documents_path_csv)
    # Check - have this folder in path
    if os.path.exists(folder_path):
        # find csv file following folder_path 
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        # if have csv file
        if csv_files:
            # Create empthy list for keep another file
            dfs = [] 
            # For loop for reading csv_file and merge them together
            for file_name in csv_files:
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                # Deleted useless column
                df = df.drop(columns=["ข้อเสนอแนะ", "ข้อเสนอแนะเพิ่มเติม"], errors='ignore')
                dfs.append(df)  # Add dataframe in list
            combined_df = pd.concat(dfs, ignore_index=True)  # Include all data
            return combined_df
    st.error(f"Please run RPA.")
    return pd.DataFrame()

# Func - For reset RPA state
def reset_rpa_state():
    st.session_state["rpa_results"] = []

# Func - DATE for GET tr & td
def get_column_day(selected_date):
    weekday = selected_date.weekday()
    weekday = (weekday + 1) % 7
    column = (weekday % 7) + 1
    day = selected_date.day
    return day, column

def get_row_day(day, column, selected_date):
    first_day_of_month = selected_date.replace(day=1)
    first_day_column = ((first_day_of_month.weekday() + 1) % 7) + 1
    day_position = day + (first_day_column - 1)
    row = (day_position - 1) // 7 + 1
    return row

# fday , fcolumn , frow
selected_fday = st.date_input("เลือกวันแรก", datetime.date.today())
fday, fcolumn = get_column_day(selected_fday)
frow = get_row_day(fday, fcolumn, selected_fday)

# lday , lcolumn , lrow
selected_lday = st.date_input("เลือกวันสุดท้าย", datetime.date.today())
lday, lcolumn = get_column_day(selected_lday)
lrow = get_row_day(lday, lcolumn, selected_lday)

st.write(f"คุณเลือกวันที่: {selected_fday} ถึงวันที่ {selected_lday}")
st.write(f"ตำแหน่งในตาราง วันแรก: วันที่ {fday}, แถวที่ {frow}, คอลัมน์ {fcolumn}")
st.write(f"ตำแหน่งในตาราง วันสุดท้าย: วันที่ {lday}, แถวที่ {lrow}, คอลัมน์ {lcolumn}")

period_day = (selected_lday.day - selected_fday.day) + 1

st.write(f"ระยะวันทั้งหมด {period_day}")

if "rpa_dataframe" not in st.session_state:
    st.session_state["rpa_dataframe"] = pd.DataFrame(columns=["Filename", "Code", "Class Predict", "Confidence"])
if "rpa_results" not in st.session_state:
    st.session_state["rpa_results"] = []

torch.backends.cudnn.benchmark = True  # เพิ่มประสิทธิภาพเมื่อใช้ GPU
torch.backends.cudnn.enabled = True

# Func - Load YOLOv8 Model only once
def load_model():
    model_path = "best11_50_8.pt"
    model = YOLO(model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu") # ส่งโมเดลไปที่ GPU ถ้ามี
    st.sidebar.write(f"YOLO is running on: {next(model.parameters()).device}")
    return model

try:
    st.sidebar.write("Loading YOLOv8 model...")
    model = load_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()


# Func - For process pull images
# This Func for pull images many folder
def process_image_RPA(uploaded_file):
    try:
        # Open the uploaded image and convert to RGB
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_image = original_image.resize((640, 640))
        image_array = np.array(resized_image)

        # Process images by YOLO Model
        results = model.predict(image_array)
        detections = results[0].boxes  # Get bounding boxes
        rendered_image = results[0].plot()  # Render detections on the image
        detected_image = Image.fromarray(rendered_image)

        # Prepare detection info
        detection_info = []
        class_count = {}
        max_confidence = {}

        # Collect detected classes and their confidences
        if detections is not None:
            for box in detections:
                class_name = results[0].names[int(box.cls)]  # Get class name
                confidence = box.conf.item() * 100  # Confidence as percentage
                detection_info.append((class_name, confidence))
                
                # Count occurrences of each class and store max confidence
                if class_name not in class_count:
                    class_count[class_name] = 0
                    max_confidence[class_name] = 0
                class_count[class_name] += 1
                max_confidence[class_name] = max(max_confidence[class_name], confidence)

        # Handle cases where detection_info is empty
        if not detection_info:
            final_class = "check"
            final_confidence = 0
        else:
            # Determine final class based on conditions
            detected_classes = set(class_count.keys())
            if "correct" in detected_classes and len(detected_classes) > 1:
                # Condition 1: If "correct" exists with other classes
                final_class = "check"
                final_confidence = max_confidence["correct"]
            elif "correct" in detected_classes and max_confidence["correct"] < 80:
                # New Condition: If "correct" confidence is less than 80%
                final_class = "check"
                final_confidence = max_confidence["correct"]
            elif "incorrect" in detected_classes and max_confidence["incorrect"] < 80:
                # New Condition: If "correct" confidence is less than 80%
                final_class = "check"
                final_confidence = max_confidence["incorrect"]
            elif len(detected_classes) > 1:
                # Condition 2: If there is no "correct" but multiple classes exist
                final_class = "incorrect"
                final_confidence = max(max_confidence.values())
            else:
                # Condition 3: If all bounding boxes are the same class
                final_class = list(detected_classes)[0]
                final_confidence = max_confidence[final_class]

        # Return processed data
        return detected_image, [(final_class, final_confidence)]
    except Exception as e:
        st.error(f"Error processing image {uploaded_file.name}: {e}")
        return None, None
    
documents_path = os.path.join(os.path.expanduser("~"), "Documents", "YOLOAppData")

# Part - For process RPA just click
if st.button("RPA"):
    # documents_path = os.path.dirname(os.path.abspath(__file__))
    # Folder name for str date
    selected_folder_name = selected_fday.strftime("%Y-%m-%d")
    # Var image & csv FOLDER
    # folder branch code ??
    image_folder = os.path.join(documents_path, "IMAGE_file")
    csv_folder = os.path.join(documents_path, "CSV_file" , f"{selected_folder_name}_csv")

    # Reset value by session_state Before display again
    if "rpa_results" in st.session_state:
        st.session_state["rpa_results"] = []  
    if "rpa_dataframe" in st.session_state:
        st.session_state["rpa_dataframe"] = pd.DataFrame()  

    # call using RPA_file that process it by subprocess
    if not (os.path.exists(image_folder) or os.path.exists(csv_folder)):
        try:
            st.sidebar.write("Running RPA script to fetch images...")
            result = subprocess.run([
                "python", "rpa.edit.py", str(frow), str(fcolumn), str(), 
                str(selected_fday.month), str(selected_fday.year),str(period_day)
            ], capture_output=True, text=True, encoding="utf-8")
            
            if result.returncode == 0:
                st.sidebar.success("RPA script completed successfully!")
            else:
                st.error(f"RPA script failed. Error: {result.stderr}")
                st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.stop()
    
    # Load data(csv) 
    dataframe = load_csv_data()
    
    if os.path.exists(image_folder):
        # os.walk for go to all-folder
        folder_paths = []
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if file.endswith(".jpg"):  # เลือกเฉพาะไฟล์รูปภาพ
                    folder_paths.append(os.path.join(root, file))

        # Check - Have picture in folder or not ?
        if folder_paths:
            st.session_state["rpa_results"] = []
            st.session_state["rpa_dataframe"] = pd.DataFrame()

            for image_file in folder_paths:
                # Pull name-branch-code
                filename = os.path.splitext(os.path.basename(image_file))[0]
                code = os.path.basename(os.path.dirname(image_file))

                # Call function process image RPA
                detected_image, detection_info = process_image_RPA(image_file) 

                # Save value in session_state
                if detected_image and isinstance(detection_info, list) and detection_info:
                    st.session_state["rpa_results"].append({
                        "Filename": filename,
                        "Code": code,
                        "Image File": image_file,
                        "Detection Info": detection_info
                    })

                    # Add data in DataFrame
                    for cls, confidence in detection_info:
                        new_row = pd.DataFrame([{
                            "Filename": filename,
                            "Code": code,
                            "Class Predict": cls,
                            "Confidence": confidence
                        }])
                        st.session_state["rpa_dataframe"] = pd.concat([st.session_state["rpa_dataframe"], new_row], ignore_index=True)
        
    st.write("RPA process completed. Data is ready for viewing.")
    
# Display PART 
if not st.session_state["rpa_dataframe"].empty:
    dataframe = st.session_state["rpa_dataframe"]
    csv_data = load_csv_data()
    
    if not dataframe.empty and not csv_data.empty:
        merged_data = pd.merge(csv_data, dataframe, left_on="รหัสร้าน", right_on="Code", how="outer").drop(columns=["Code"])
        
        # Add DROP-DOWN for select zone
        unique_zones = ["ALL"] + sorted(merged_data["โซน"].dropna().unique().tolist())
        selected_zone = st.selectbox("Select Zone", unique_zones)
        
        # Filter data(zone) from zone that user select
        filtered_data = merged_data.copy()
        if selected_zone != "ALL":
            filtered_data = filtered_data[filtered_data["โซน"] == selected_zone]
        
        # Filter data(branch-code) from branch-code that user select 1
        unique_codes = ["ALL"] + sorted(filtered_data["รหัสร้าน"].dropna().unique().tolist())
        selected_code = st.selectbox("Select Code", unique_codes)
        
        # Filter data(branch-code) from branch-code that user select 2
        if selected_code != "ALL":
            filtered_data = filtered_data[filtered_data["รหัสร้าน"] == selected_code]
        
        # Update task Class Predict by branch-code that user select
        unique_classes = ["ALL"] + sorted(filtered_data["Class Predict"].dropna().unique().tolist())
        selected_class = st.selectbox("Select Class", unique_classes)
        
        # Filter by Class Predict that user choose
        if selected_class != "ALL":
            filtered_data = filtered_data[filtered_data["Class Predict"] == selected_class]
        
        # FOR WHAT ???
        st.dataframe(filtered_data.reset_index(drop=True))

        # FOR WHAT ???
        if not filtered_data.empty:
            st.write("## Classification Distribution")
            class_counts = filtered_data["Class Predict"].value_counts()

            fig, ax = plt.subplots()

            if selected_class != "ALL" and selected_class in class_counts:
                # If choose a class to display Full-Pie Chart and display text center
                percentage = (class_counts[selected_class] / len(filtered_data)) * 100
                ax.pie([1], labels=[""], startangle=90, colors=["#99c2ff"])  # hide label
                ax.text(0, 0, f"{selected_class}\n{percentage:.1f}%", ha="center", va="center", fontsize=14)
            else:
                # If choose "ALL" to display Normal-Pie Chart
                ax.pie((class_counts / len(filtered_data)) * 100, labels=class_counts.index, autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)

            ax.axis("equal")
            st.pyplot(fig)


        for result in st.session_state["rpa_results"]:
            # Check result["Code"] in the selected zone
            if selected_zone != "ALL":
                if result["Code"] not in filtered_data["รหัสร้าน"].values:
                    continue

            # Filter by branch-code
            if selected_code != "ALL" and result["Code"] != selected_code:
                continue

            # Filter type detected image
            if selected_class != "ALL" and not any(cls == selected_class for cls, _ in result["Detection Info"]):
                continue
            st.markdown(f"#### รหัสร้าน: {result['Code']}")
            st.markdown(f"#### Detected Image: {result['Filename']}")
            st.image(result['Image File'], use_container_width=True)
            detection_text = "<br>".join([f"{cls}" for cls, _ in result["Detection Info"]])
            additional_text = {
                "correct": "Your PM work image meets the standard.",
                "check": "Your PM work image is under review. Multiple types detected.",
                "incorrect": "Your PM work image doesn't meet the standard. Please check for cleanliness.",
                "fail": "Your PM work image doesn't meet the standard. Please check for cleanliness.",
                "undetected": "No detectable objects found in the image. Please recheck the image."
            }.get(detection_text, "Unknown status")
            st.markdown(
                f'<div style="border: 2px solid black; padding: 10px; background-color: #f0f0f0; text-align: center;">'
                f'<h2 style="color: black">{detection_text}</h2>'
                f'<p style="color: black">{additional_text}</p>'
                f'</div><br><br>', unsafe_allow_html=True
            )

st.markdown("<div class='footer'>Developed by Your Name | Contact: satit102688@gmail.com</div>", unsafe_allow_html=True)
