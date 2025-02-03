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

# ฟังก์ชันสำหรับโหลดข้อมูลจาก CSV
def load_csv_data(selected_date):
    folder_path = f"{selected_date}_csv"
    if os.path.exists(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if csv_files:
            file_path = os.path.join(folder_path, csv_files[0])  # ใช้ไฟล์ CSV แรกที่พบ
            df = pd.read_csv(file_path)
            df = df.drop(columns=["เวลาเริ่มงาน", "เวลาปิดงาน", "ข้อเสนอแนะ", "ข้อเสนอแนะเพิ่มเติม"], errors='ignore')
            return df
    st.error(f"No CSV file found in: {folder_path}")
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

selected_date = st.date_input("เลือกวันที่", datetime.date.today())
csv_data = load_csv_data(selected_date)
day, column = get_column_day(selected_date)
row = get_row_day(day, column, selected_date)

st.write(f"คุณเลือกวันที่: {selected_date}")
# st.write(f"ตำแหน่งในตาราง: วันที่ {day}, แถวที่ {row}, คอลัมน์ {column}")

if "rpa_dataframe" not in st.session_state:
    st.session_state["rpa_dataframe"] = pd.DataFrame(columns=["Filename", "Code", "Class Predict", "Confidence"])
if "rpa_results" not in st.session_state:
    st.session_state["rpa_results"] = []
if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = pd.DataFrame(columns=["Filename", "Class Predict", "Confidence"])


def reset_upload_state():
    st.session_state["rpa_results"] = []
    st.session_state["rpa_dataframe"] = pd.DataFrame(columns=["Filename", "Code", "Class Predict", "Confidence"])

def reset_rpa_state():
    st.session_state["rpa_results"] = []


# Upload Images
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def get_dataframe():
    return st.session_state["dataframe"]


# Load YOLOv8 Model only once
@st.cache_resource
def load_model():
    model_path = r'C:\selenium_web\web_detect_tank\best.pt'
    return YOLO(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    st.sidebar.write("Loading YOLOv8 model...")
    model = load_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()


# Function to process an image
def process_image(uploaded_file):
    try:
        # Open the uploaded image
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_image = original_image.resize((640, 640))
        image_array = np.array(resized_image)
        results = model.predict(image_array)  # Removed conf=confidence_threshold
        detections = results[0].boxes  # Get bounding boxes
        rendered_image = results[0].plot()  # Render detections on the image
        detected_image = Image.fromarray(rendered_image)

        # Prepare detection info
        detection_info = []
        types_detected = set()
        max_confidence = 0
        main_type = ""

        if detections is not None:
            for box in detections:
                class_name = results[0].names[int(box.cls)]  # Get class name
                confidence = box.conf.item() * 100  # Get confidence as percentage
                detection_info.append((class_name, confidence))
                types_detected.add(class_name)

                if confidence > max_confidence:
                    max_confidence = confidence
                    main_type = class_name

        # Define final type based on conditions
        if "correct" in types_detected and "incorrect" in types_detected and "fail" in types_detected:
            # Check if any class has confidence less than 90%
            if any(conf < 90 for _, conf in detection_info):
                final_type = "check"
            else:
                final_type = main_type
        elif "correct" in types_detected and len(types_detected) > 1:
            final_type = "check"
        elif len(types_detected) > 1:
            final_type = main_type
        else:
            final_type = main_type if main_type else "check"

        # Update dataframe with filename, class prediction, and confidence
        dataframe = get_dataframe()
        dataframe.loc[len(dataframe)] = [uploaded_file.name, final_type, max_confidence]
        st.session_state["dataframe"] = dataframe

        return detected_image, detection_info
    except Exception as e:
        st.error(f"Error processing image {uploaded_file.name}: {e}")
        return None, None

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
            elif "correct" in detected_classes and max_confidence["correct"] < 60:
                # New Condition: If "correct" confidence is less than 80%
                final_class = "check"
                final_confidence = max_confidence["correct"]
            elif "incorrect" in detected_classes and max_confidence["incorrect"] < 60:
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

# Automatically process uploaded images
if uploaded_files:
    reset_rpa_state()
    for uploaded_file in uploaded_files:
        process_image(uploaded_file)

    # Sidebar toggle for viewing options
    view_option = st.sidebar.radio("View Option", ("Show DataFrame", "Show Images"))

    # Show DataFrame
    if view_option == "Show DataFrame":
        # Display the DataFrame with results
        st.write("### Detection Results")
        dataframe = get_dataframe()
        
        # Remove file extensions from filenames
        dataframe["Filename"] = dataframe["Filename"].str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)

        st.dataframe(dataframe)

        # Add download button for CSV
        if not dataframe.empty:
            # แสดงแถบใส่ชื่อไฟล์
            file_name_input = st.text_input("Enter file name to save (without extension):")
            
            # ถ้าใส่ชื่อไฟล์แล้ว
            if file_name_input:
                csv = dataframe.to_csv(index=False, header=True).encode('utf-8')
                st.download_button(
                    label="Download Data as CSV",
                    data=csv,
                    file_name=f"{file_name_input}.csv",
                    mime="text/csv",
                )
            else:
                # ถ้าไม่ได้ใส่ชื่อไฟล์
                st.download_button(
                    label="Download Data as CSV",
                    data=b"",  # ส่งข้อมูลว่าง
                    disabled=True  # ทำให้ปุ่มไม่สามารถคลิกได้
                )
                st.error("Please enter a file name to save.")


        # Display Pie Chart
        if not dataframe.empty:
            st.write("### Classification Distribution")
            class_counts = dataframe["Class Predict"].value_counts()
            class_percentages = (class_counts / len(dataframe)) * 100

            fig, ax = plt.subplots()
            ax.pie(class_percentages, labels=class_counts.index, autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)
            ax.axis("equal")  # Equal aspect ratio ensures the pie chart is a circle
            st.pyplot(fig)
            st.write("#")

    # Show Images
    elif view_option == "Show Images":
        # Show all images toggle
        show_all = st.button("Show All")
        if 'show_all_state' not in st.session_state:
            st.session_state.show_all_state = False

        # Toggle between showing all images and just one
        if show_all:
            st.session_state.show_all_state = not st.session_state.show_all_state

        # Display all images or selected image based on the toggle
        if uploaded_files:
            if st.session_state.show_all_state:
                for uploaded_file in uploaded_files:
                    # Process the image
                    detected_image, detection_info = process_image(uploaded_file)

                    if detected_image:
                        st.markdown(f"#### Detected Image: {uploaded_file.name}")
                        st.image(uploaded_file, use_container_width=True)
                        # st.image(detected_image, use_container_width=True)

                        # Get classification and confidence from DataFrame
                        dataframe = get_dataframe()
                        file_data = dataframe[dataframe["Filename"] == uploaded_file.name]
                        if not file_data.empty:
                            final_type = file_data.iloc[0]["Class Predict"]
                            max_confidence = file_data.iloc[0]["Confidence"]

                            # Define additional text based on type
                        if final_type == "correct":
                            additional_text = (
                                "Your PM work image meets the standard."
                            )
                        elif final_type == "check":
                            additional_text = (
                                "Your PM work image is under review. Multiple types detected."
                            )
                        elif final_type == "incorrect":
                            additional_text = (
                                "Your PM work image doesn't meet the standard.<br>"
                                "Please check for cleanliness, there should be no residual water and no sediment."
                            )
                        elif final_type == "fail":
                            additional_text = (
                                "Your PM work image doesn't meet the standard.<br>"
                                "Please check for cleanliness, there should be no residual water and no sediment."
                            )
                        elif final_type == "undetected":
                            additional_text = (
                                "No detectable objects found in the image. Please recheck the image."
                            )
                        st.markdown(
                            f'<div style="border: 2px solid black; padding: 10px; background-color: #f0f0f0; text-align: center;">'
                            f'<h2 style="color: black">{final_type}</h2>'
                            f'<p style="color: black">{additional_text}</p>'
                            f'</div>'
                            f'<br>' 
                            f'<br>',  # เพิ่มช่องว่างหลัง div
                            unsafe_allow_html=True
                        )

                    else:
                        st.write("No detections found in the DataFrame.")
            else:
                # Process and display the selected image
                selected_image_name = st.selectbox("Select an Image to View:", [file.name for file in uploaded_files])
                selected_file = next(file for file in uploaded_files if file.name == selected_image_name)
                
                # Process the selected image
                detected_image, detection_info = process_image(selected_file)

                if detected_image:
                    st.markdown(f"#### Detected Image: {selected_file.name}")
                    st.image(selected_file, use_container_width=True)

                    # Get classification and confidence from DataFrame
                    dataframe = get_dataframe()
                    file_data = dataframe[dataframe["Filename"] == selected_file.name]
                    if not file_data.empty:
                        final_type = file_data.iloc[0]["Class Predict"]
                        max_confidence = file_data.iloc[0]["Confidence"]

                        # Define additional text based on type
                        if final_type == "correct":
                            additional_text = (
                                "Your PM work image meets the standard."
                            )
                        elif final_type == "incorrect" or "fail":
                            additional_text = (
                                "Your PM work image doesn't meet the standard.<br>"
                                "Please check for cleanliness, there should be no residual water and no sediment."
                            )
                        elif final_type == "check":
                            additional_text = (
                                "Your PM work image is under review. Multiple types detected."
                            )
                        elif final_type == "undetected":
                            additional_text = (
                                "No detectable objects found in the image. Please recheck the image."
                            )
                        st.markdown(
                            f'<div style="border: 2px solid black; padding: 10px; background-color: #f0f0f0; text-align: center;">'
                            f'<h2 style="color: black">{final_type}</h2>'
                            f'<p style="color: black">{additional_text}</p>'
                            f'</div>'
                            f'<br>' 
                            f'<br>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("No detections found in the DataFrame.")


if st.button("RPA"):
    reset_upload_state()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    selected_folder_name = selected_date.strftime("%Y-%m-%d")
    image_folder = os.path.join(script_dir, selected_folder_name)
    csv_folder = os.path.join(script_dir, f"{selected_folder_name}_csv")
    
    if not (os.path.exists(image_folder) and os.path.exists(csv_folder)):
        try:
            st.sidebar.write("Running RPA script to fetch images...")
            result = subprocess.run([
                "python", "rpa_6.py", str(row), str(column), str(selected_date), 
                str(selected_date.month), str(selected_date.year)
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
    dataframe = load_csv_data(selected_date)
    
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
    csv_data = load_csv_data(selected_date)
    
    if not dataframe.empty and not csv_data.empty:
        merged_data = pd.merge(csv_data, dataframe, left_on="รหัสร้าน", right_on="Code", how="outer").drop(columns=["Code"])
        selected_code = st.selectbox("Select Code", ["ALL"] + list(merged_data["รหัสร้าน"].dropna().unique()))
        
        filtered_dataframe = merged_data if selected_code == "ALL" else merged_data[merged_data["รหัสร้าน"] == selected_code]
        selected_class = st.selectbox("Select Class", ["ALL"] + list(filtered_dataframe["Class Predict"].dropna().unique()))
        if selected_class != "ALL":
            filtered_dataframe = filtered_dataframe[filtered_dataframe["Class Predict"] == selected_class]
        
        st.dataframe(filtered_dataframe.reset_index(drop=True))
        
        if not filtered_dataframe.empty:
            st.write("## Classification Distribution")
            class_counts = filtered_dataframe["Class Predict"].value_counts()
            fig, ax = plt.subplots()
            ax.pie((class_counts / len(filtered_dataframe)) * 100, labels=class_counts.index, autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)
            ax.axis("equal")
            st.pyplot(fig)
        
        for result in st.session_state["rpa_results"]:
            if selected_code != "ALL" and result["Code"] != selected_code:
                continue
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
