import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile, cv2, os

st.title("YOLOv8 Object Detection")
conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
source_type = st.radio("Input Type", ("Image", "Video"))

model = YOLO("YOLOv8trained model/best.pt") 


if source_type == "Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Get extension of uploaded file
        suffix = os.path.splitext(uploaded_file.name)[1]  

        # Create temp file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image.save(tmp.name)
            results = model.predict(source=tmp.name, conf=conf)

        result_img = results[0].plot()
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detection", use_column_width=True)

else:
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4","avi","mov","mkv"])
    if uploaded_video:
        # Keep original extension
        suffix = os.path.splitext(uploaded_video.name)[1]  

        # Create temp file with correct extension
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_video.read())
        tfile.close()  

        st.video(tfile.name)
        results = model.predict(source=tfile.name, conf=conf, save=True)

        output_dir = results[0].save_dir
        output_files = [f for f in os.listdir(output_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if output_files:
            st.video(os.path.join(output_dir, output_files[0]))
