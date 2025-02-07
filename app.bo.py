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

documents_path = os.path.join(os.path.expanduser("~"), "Documents", "YOLOAppData")
# ฟังก์ชันสำหรับโหลดข้อมูลจาก CSV
def load_csv_data(selected_date):
    folder_path = os.path.join(documents_path, f"{selected_date}_csv")
    if os.path.exists(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if csv_files:
            file_path = os.path.join(folder_path, csv_files[0])  # ใช้ไฟล์ CSV แรกที่พบ
            df = pd.read_csv(file_path)
            df = df.drop(columns=["ข้อเสนอแนะ", "ข้อเสนอแนะเพิ่มเติม"], errors='ignore')
            return df
    # st.error(f"No CSV file found in: {folder_path}")
    st.error(f"Please run RPA.")
    return pd.DataFrame()

# Date
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

# selected_date = st.date_input("เลือกวันที่", datetime.date.today())
# csv_data = load_csv_data(selected_date)
# day, column = get_column_day(selected_date)
# row = get_row_day(day, column, selected_date)

selected_fday = st.date_input("เลือกวันแรก", datetime.date.today())
csv_data = load_csv_data(selected_fday)
day, column = get_column_day(selected_fday)
row = get_row_day(day, column, selected_fday)

selected_lday = st.date_input("เลือกวันสุดท้าย", datetime.date.today())
csv_data = load_csv_data(selected_lday)
day, column = get_column_day(selected_lday)
row = get_row_day(day, column, selected_lday)

st.write(f"คุณเลือกวันที่: {selected_fday} ถึงวันที่ {selected_lday}")
# st.write(f"ตำแหน่งในตาราง: วันที่ {day}, แถวที่ {row}, คอลัมน์ {column}")

period_day = selected_lday.day - selected_fday.day

st.write(f"ระยะวันทั้งหมด {period_day}")

if "rpa_dataframe" not in st.session_state:
    st.session_state["rpa_dataframe"] = pd.DataFrame(columns=["Filename", "Code", "Class Predict", "Confidence"])
if "rpa_results" not in st.session_state:
    st.session_state["rpa_results"] = []


def reset_rpa_state():
    st.session_state["rpa_results"] = []

torch.backends.cudnn.benchmark = True  # เพิ่มประสิทธิภาพเมื่อใช้ GPU
torch.backends.cudnn.enabled = True


# Load YOLOv8 Model only once
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


# Function to process an image
def process_image_RPA(uploaded_file):
    try:
        # Open the uploaded image and convert to RGB
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_image = original_image.resize((640, 640))
        image_array = np.array(resized_image)
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
    
    # How to get folder for display

if st.button("RPA"):
    # documents_path = os.path.dirname(os.path.abspath(__file__))
    selected_folder_name = selected_fday.strftime("%Y-%m-%d")
    image_folder = os.path.join(documents_path, selected_folder_name)
    csv_folder = os.path.join(documents_path, f"{selected_folder_name}_csv")

    # รีเซ็ตค่าใน session_state ก่อนการแสดงผลใหม่
    if "rpa_results" in st.session_state:
        st.session_state["rpa_results"] = []  # ลบข้อมูลเก่าที่เก็บไว้
    if "rpa_dataframe" in st.session_state:
        st.session_state["rpa_dataframe"] = pd.DataFrame()  # ลบข้อมูลเก่าใน DataFrame
    
    # if not (os.path.exists(image_folder) and os.path.exists(csv_folder)):
    #     try:
    #         st.sidebar.write("Running RPA script to fetch images...")
    #         result = subprocess.run([
    #             "python", "rpa.py", str(row), str(column), str(selected_fday), 
    #             str(selected_fday.month), str(selected_fday.year)
    #         ], capture_output=True, text=True)
            
    #         if result.returncode == 0:
    #             st.sidebar.success("RPA script completed successfully!")
    #         else:
    #             st.error(f"RPA script failed. Error: {result.stderr}")
    #             st.stop()
    #     except Exception as e:
    #         st.error(f"An unexpected error occurred: {e}")
    #         st.stop()

    if not (os.path.exists(image_folder) and os.path.exists(csv_folder)):
        try:
            st.sidebar.write("Running RPA script to fetch images...")
            result = subprocess.run([
                "python", "rpa.edit.py", str(row), str(column), str(), 
                str(selected_fday.month), str(selected_fday.year),str(period_day)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                st.sidebar.success("RPA script completed successfully!")
            else:
                st.error(f"RPA script failed. Error: {result.stderr}")
                st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.stop()
    
    # โหลดข้อมูลหลังจากรัน RPA script
    dataframe = load_csv_data(selected_fday)
    
    if os.path.exists(image_folder):
        image_files = [os.path.join(root, file) for root, _, files in os.walk(image_folder) for file in files if file.endswith(".jpg")]
        reset_rpa_state()
        
        if image_files:
            for image_file in image_files:
                filename = os.path.splitext(os.path.basename(image_file))[0]
                code = os.path.basename(os.path.dirname(image_file))
                detected_image, detection_info = process_image_RPA(image_file)
                
                if detected_image and isinstance(detection_info, list) and detection_info:
                    st.session_state["rpa_results"].append({"Filename": filename, "Code": code, "Image File": image_file, "Detection Info": detection_info})
                    for cls, confidence in detection_info:
                        new_row = pd.DataFrame([{ "Filename": filename, "Code": code, "Class Predict": cls, "Confidence": confidence }])
                        st.session_state["rpa_dataframe"] = pd.concat([st.session_state["rpa_dataframe"], new_row], ignore_index=True)
    
    st.write("RPA process completed. Data is ready for viewing.")
    
# แสดงผลลัพธ์
if not st.session_state["rpa_dataframe"].empty:
    dataframe = st.session_state["rpa_dataframe"]
    csv_data = load_csv_data(selected_fday)
    
    if not dataframe.empty and not csv_data.empty:
        merged_data = pd.merge(csv_data, dataframe, left_on="รหัสร้าน", right_on="Code", how="outer").drop(columns=["Code"])
        
        # เพิ่ม dropdown สำหรับเลือกโซนที่จุดนี้
        unique_zones = ["ALL"] + sorted(merged_data["โซน"].dropna().unique().tolist())
        selected_zone = st.selectbox("Select Zone", unique_zones)
        
        # กรองข้อมูลตามโซนที่เลือก
        filtered_data = merged_data.copy()
        if selected_zone != "ALL":
            filtered_data = filtered_data[filtered_data["โซน"] == selected_zone]
        
        # อัปเดตรายการรหัสร้านตามโซนที่เลือก
        unique_codes = ["ALL"] + sorted(filtered_data["รหัสร้าน"].dropna().unique().tolist())
        selected_code = st.selectbox("Select Code", unique_codes)
        
        # กรองข้อมูลตามรหัสร้านที่เลือก
        if selected_code != "ALL":
            filtered_data = filtered_data[filtered_data["รหัสร้าน"] == selected_code]
        
        # อัปเดตรายการ Class Predict ตามรหัสร้านที่เลือก
        unique_classes = ["ALL"] + sorted(filtered_data["Class Predict"].dropna().unique().tolist())
        selected_class = st.selectbox("Select Class", unique_classes)
        
        # กรองข้อมูลตาม Class Predict ที่เลือก
        if selected_class != "ALL":
            filtered_data = filtered_data[filtered_data["Class Predict"] == selected_class]
        
        st.dataframe(filtered_data.reset_index(drop=True))

        if not filtered_data.empty:
            st.write("## Classification Distribution")
            class_counts = filtered_data["Class Predict"].value_counts()

            fig, ax = plt.subplots()

            if selected_class != "ALL" and selected_class in class_counts:
                # ถ้าเลือก class เดียวให้แสดง Pie Chart เต็มวง และแสดงข้อความตรงกลาง
                percentage = (class_counts[selected_class] / len(filtered_data)) * 100
                ax.pie([1], labels=[""], startangle=90, colors=["#99c2ff"])  # ซ่อน label
                ax.text(0, 0, f"{selected_class}\n{percentage:.1f}%", ha="center", va="center", fontsize=14)
            else:
                # ถ้าเลือก "ALL" ให้แสดง Pie Chart ปกติ
                ax.pie((class_counts / len(filtered_data)) * 100, labels=class_counts.index, autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)

            ax.axis("equal")
            st.pyplot(fig)


        for result in st.session_state["rpa_results"]:
            # ตรวจสอบว่า result["Code"] อยู่ในโซนที่เลือก
            if selected_zone != "ALL":
                if result["Code"] not in filtered_data["รหัสร้าน"].values:
                    continue

            # กรองตามรหัสร้าน
            if selected_code != "ALL" and result["Code"] != selected_code:
                continue

            # กรองตามประเภทการตรวจจับ
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
