from src.utils import (
    calculate_ear,
    calculate_fatigue_score,
    get_recommendation
)
import streamlit as st
import cv2
import mediapipe as mp
import time
import joblib

import av
import threading
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fatigue_model.pkl"
)


ARCHITECTURE_IMG = os.path.join(
    BASE_DIR,
    "assets",
    "architecture.png"
)

ATTENTIVE_IMG = os.path.join(
    BASE_DIR,
    "screenshots",
    "attentive_demo.jpg"
)

DROWSY_IMG = os.path.join(
    BASE_DIR,
    "screenshots",
    "drowsy_demo.jpg"
)


# PAGE CONFIG


st.set_page_config(
    page_title="DriverGuard AI",
    page_icon=" ",
    layout="wide"
)


st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.metric-card {
    background-color: #1E2635;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

.big-title {
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    color: gray;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


st.sidebar.title(" DriverGuard AI")

st.sidebar.markdown("---")

start_monitoring = st.sidebar.button(
    "▶ Start Monitoring",
    use_container_width=True
)

st.sidebar.success("🟢 Camera Connected")

st.sidebar.info(" Fatigue Model Loaded")

st.sidebar.markdown("---")

st.sidebar.write("Version 1.0")

st.sidebar.write("AI Internship Project")


st.markdown(
    '<div class="big-title">🚗 DriverGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-Time Driver Fatigue Monitoring System</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================
# LANDING PAGE
# =========================

if not start_monitoring:

    st.header("About DriverGuard AI")

    st.write("""
    DriverGuard AI is a real-time fatigue detection system that uses
    Computer Vision and Machine Learning to monitor driver alertness.

    The system continuously analyzes eye behaviour and predicts whether
    the driver is focused or drowsy.
    """)

    st.subheader("Features")

    st.markdown("""
    - Real-time Eye Tracking
    - Eye Aspect Ratio (EAR)
    - Blink Rate Monitoring
    - Eye Closure Duration
    - PERCLOS Calculation
    - Random Forest Prediction
    - Live Dashboard
    """)

    st.subheader("System Architecture")

    st.image(
        ARCHITECTURE_IMG,
        use_container_width=True
    )

    st.subheader("Demo Screenshots")

    st.subheader("📸 Demo Screenshots")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            ATTENTIVE_IMG,
            caption="Focused Driver"
        )

    with col2:
        st.image(
            DROWSY_IMG,
            caption="Drowsy Driver"
        )

    st.stop()


print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)
print("EXISTS =", os.path.exists(MODEL_PATH))

model = joblib.load(MODEL_PATH)


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]


class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

        self.blink_count = 0
        self.closed_frames = 0
        self.closed_eye_frames = 0
        self.total_frames = 0

        self.eye_closed_start = None
        self.session_start = time.time()

        self.EAR_THRESHOLD = 0.20

        self.status = "FOCUSED"
        self.confidence = 100
        self.fatigue_score = 0

        self.ear = 0
        self.blink_rate = 0
        self.closure_duration = 0
        self.perclos = 0

        self.recommendation = "Drive Safely"

        self.lock = threading.Lock()

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        self.total_frames += 1

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:

            face = results.multi_face_landmarks[0]

            h, w, _ = img.shape

            eye_points = []

            for idx in LEFT_EYE:

                landmark = face.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                eye_points.append((x, y))

                cv2.circle(
                    img,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

            self.ear = calculate_ear(eye_points)

            if self.ear < self.EAR_THRESHOLD:

                self.closed_frames += 1
                self.closed_eye_frames += 1

                if self.eye_closed_start is None:
                    self.eye_closed_start = time.time()

                self.closure_duration = (
                    time.time() - self.eye_closed_start
                )

            else:

                if self.closed_frames >= 2:
                    self.blink_count += 1

                self.closed_frames = 0
                self.eye_closed_start = None
                self.closure_duration = 0

            elapsed_minutes = (
                time.time() - self.session_start
            ) / 60

            if elapsed_minutes > 0:

                self.blink_rate = (
                    self.blink_count /
                    elapsed_minutes
                )

            self.perclos = (
                self.closed_eye_frames /
                self.total_frames
            ) * 100

            features = [[
                self.ear,
                self.blink_rate,
                self.closure_duration,
                self.perclos
            ]]

            prediction = model.predict(features)[0]

            probability = max(
                model.predict_proba(features)[0]
            )

            self.confidence = round(
                probability * 100,
                2
            )

            if prediction == 1:
                self.status = "DROWSY"
                color = (0, 0, 255)
            else:
                self.status = "FOCUSED"
                color = (0, 255, 0)

            self.fatigue_score = calculate_fatigue_score(
                self.ear,
                self.perclos,
                self.closure_duration
            )

            self.recommendation = get_recommendation(
                self.fatigue_score
            )

            cv2.putText(
                img,
                self.status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            cv2.putText(
                img,
                f"EAR : {self.ear:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                img,
                f"Fatigue : {self.fatigue_score:.1f}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                img,
                f"Confidence : {self.confidence:.1f}%",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =========================
# WEBRTC CONFIG
# =========================


rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

# =========================
# DASHBOARD LAYOUT
# =========================

left_col, right_col = st.columns([2, 1])

with left_col:

    ctx = webrtc_streamer(
        key="driverguard",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

with right_col:

    status_placeholder = st.empty()

    fatigue_placeholder = st.empty()

    confidence_placeholder = st.empty()

st.divider()

e1, e2, e3, e4 = st.columns(4)

ear_placeholder = e1.empty()

blink_placeholder = e2.empty()

closure_placeholder = e3.empty()

perclos_placeholder = e4.empty()

recommendation_placeholder = st.empty()

if ctx and ctx.video_processor:

    processor = ctx.video_processor

    status = processor.status

    if status == "ALERT":

        status_placeholder.success(
            f"🟢 {status}"
        )

    else:

        status_placeholder.error(
            f"🔴 {status}"
        )

    fatigue_placeholder.metric(
        "Fatigue Score",
        f"{processor.fatigue_score}%"
    )

    confidence_placeholder.metric(
        "Confidence",
        f"{processor.confidence}%"
    )

    ear_placeholder.metric(
        "EAR",
        f"{processor.ear:.2f}"
    )

    blink_placeholder.metric(
        "Blink Rate",
        f"{processor.blink_rate:.1f}/min"
    )

    closure_placeholder.metric(
        "Eye Closure",
        f"{processor.closure_duration:.2f}s"
    )

    perclos_placeholder.metric(
        "PERCLOS",
        f"{processor.perclos:.1f}%"
    )

    recommendation_placeholder.warning(
        processor.recommendation
    )

else:

    st.info(
        "Click START above the camera feed."
    )
