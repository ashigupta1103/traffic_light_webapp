import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("🚦 Real-time Traffic Light Detection System")

uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with open("input_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("input_video.mp4")

    cap = cv2.VideoCapture("input_video.mp4")

    ret, frame = cap.read()
    if not ret:
        st.error("❌ Could not read video.")
        cap.release()
    else:
        st.subheader("First Frame")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        lower_red1, upper_red1 = (0, 100, 100), (10, 255, 255)
        lower_red2, upper_red2 = (160, 100, 100), (180, 255, 255)
        lower_yellow, upper_yellow = (20, 100, 100), (30, 255, 255)
        lower_green, upper_green = (40, 100, 100), (90, 255, 255)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output_stable.mp4", fourcc, 30,
                              (int(cap.get(3)), int(cap.get(4))))

        lights_over_time = []

        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            red_count = cv2.countNonZero(mask_red)
            yellow_count = cv2.countNonZero(mask_yellow)
            green_count = cv2.countNonZero(mask_green)

            # Decide state
            if red_count > yellow_count and red_count > green_count:
                state, color = "🔴 Red Light is ON", (0, 0, 255)
            elif yellow_count > red_count and yellow_count > green_count:
                state, color = "🟡 Yellow Light is ON", (0, 255, 255)
            elif green_count > red_count and green_count > yellow_count:
                state, color = "🟢 Green Light is ON", (0, 255, 0)
            else:
                state, color = "⚠️ No clear light is visible", (255, 255, 255)

            lights_over_time.append(state)

            cv2.putText(frame, state, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            out.write(frame)

            progress.progress(min(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / frame_count, 1.0))

        cap.release()
        out.release()

        st.success("✅ Processing complete!")
        st.subheader("Processed Video")
        st.video("output_stable.mp4")

        st.subheader("Detected States Over Time")
        st.write(lights_over_time)
