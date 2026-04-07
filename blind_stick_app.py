import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Smart Blind Stick - AI Vision System",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .emergency-btn {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
        cursor: pointer;
        transition: transform 0.3s;
    }
    .emergency-btn:hover {
        transform: scale(1.02);
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .person-alert {
        background: #ff4444;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        animation: pulse 1s infinite;
        color: white;
        font-weight: bold;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    .detection-card {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .danger-card {
        border-left-color: #ff4444;
        background: #ffe0e0;
    }
    .warning-card {
        border-left-color: #ff9800;
        background: #fff0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'emergency_mode' not in st.session_state:
    st.session_state.emergency_mode = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load YOLO model with caching
@st.cache_resource
def load_model():
    """Load YOLOv8 model"""
    try:
        from ultralytics import YOLOR
        # Try different model names
        try:
            model = YOLO('yolov8n.pt')
        except:
            try:
                model = YOLO('yolov8nano.pt')
            except:
                # Fallback to a smaller model
                model = YOLO('yolov8s.pt')
        return model
    except Exception as e:
        st.error(f"⚠️ Model loading error: {e}")
        return None

def get_distance_from_bbox(bbox_height, img_height):
    """Estimate distance based on bounding box height"""
    ratio = bbox_height / img_height
    if ratio > 0.5:
        return "very close"
    elif ratio > 0.3:
        return "close"
    elif ratio > 0.15:
        return "medium"
    else:
        return "far"

def get_direction(center_x, img_width):
    """Determine direction based on x-center"""
    if center_x < img_width * 0.3:
        return "on your left"
    elif center_x > img_width * 0.7:
        return "on your right"
    else:
        return "straight ahead"

def process_image(image, model):
    """Process image and return detections"""
    if model is None:
        return None, [], [], []
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Run inference
    results = model(img_array)
    
    detections = []
    persons = []
    vehicles = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calculate metrics
                bbox_height = y2 - y1
                img_height = img_array.shape[0]
                img_width = img_array.shape[1]
                center_x = (x1 + x2) / 2
                
                distance = get_distance_from_bbox(bbox_height, img_height)
                direction = get_direction(center_x, img_width)
                
                detection = {
                    'class': name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'distance': distance,
                    'direction': direction,
                    'timestamp': datetime.now()
                }
                detections.append(detection)
                
                if name == 'person':
                    persons.append(detection)
                elif name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    vehicles.append(detection)
    
    # Get annotated image
    annotated = results[0].plot()
    
    return annotated, detections, persons, vehicles

def create_confidence_gauge(confidence):
    """Create gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Avg Detection Confidence"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': "#ff4444"},
                {'range': [50, 80], 'color': "#ff9800"},
                {'range': [80, 100], 'color': "#4caf50"}
            ]
        }
    ))
    fig.update_layout(height=250)
    return fig

def create_timeline_chart():
    """Create timeline of detections"""
    if not st.session_state.detection_history:
        return None
    
    df = pd.DataFrame(st.session_state.detection_history[-20:])
    df['time'] = pd.to_datetime(df['timestamp'])
    
    fig = px.line(df, x='time', y='confidence', color='class',
                  title="Detection Timeline",
                  labels={'confidence': 'Confidence', 'time': 'Time'})
    fig.update_layout(height=300)
    return fig

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦯 Smart Blind Stick - AI Vision System</h1>
        <p>Real-time Person & Obstacle Detection | Emergency Alert System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI Model (first time may take a moment)..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load AI model. Please check your internet connection and try again.")
        st.stop()
    
    if not st.session_state.model_loaded:
        st.success("✅ AI Model Loaded Successfully!")
        st.session_state.model_loaded = True
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2097/2097155.png", width=100)
        st.markdown("## 🎯 Controls")
        
        # Emergency button
        st.markdown("### 🚨 Emergency")
        if st.button("🔴 ACTIVATE EMERGENCY ALERT 🔴", use_container_width=True):
            st.session_state.emergency_mode = True
            st.markdown("""
            <div class="emergency-btn">
                🚨 EMERGENCY ACTIVATED! HELP NEEDED IMMEDIATELY! 🚨
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.detection_history.append({
                'timestamp': datetime.now(),
                'type': 'emergency',
                'class': 'EMERGENCY',
                'confidence': 100,
                'message': 'Emergency alert triggered'
            })
            
            # JavaScript for browser alert
            st.markdown("""
            <script>
                alert("🚨 EMERGENCY ALERT! Please seek immediate help! 🚨");
                var audio = new Audio('https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3');
                audio.play();
            </script>
            """, unsafe_allow_html=True)
        
        if st.session_state.emergency_mode:
            if st.button("🟢 Clear Emergency", use_container_width=True):
                st.session_state.emergency_mode = False
                st.success("Emergency mode cleared")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("## 📊 Statistics")
        total_detections = len([d for d in st.session_state.detection_history if d.get('type') != 'emergency'])
        person_detections = len([d for d in st.session_state.detection_history if d.get('class') == 'person'])
        emergency_count = len([d for d in st.session_state.detection_history if d.get('type') == 'emergency'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Detections", total_detections)
            st.metric("Emergency Alerts", emergency_count)
        with col2:
            st.metric("Persons Detected", person_detections)
        
        st.markdown("---")
        
        # About
        st.markdown("""
        ## ℹ️ About
        **Smart Blind Stick** uses AI to detect:
        - 👤 Persons (with distance & direction)
        - 🚗 Vehicles
        - 🚦 Obstacles
        
        **Features:**
        - Real-time detection
        - Distance estimation
        - Direction tracking
        - Emergency alerts
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Image Input")
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload Image", "📸 Camera", "🔗 Image URL"],
            horizontal=True
        )
        
        image = None
        
        if input_method == "📁 Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", 
                                            type=['jpg', 'jpeg', 'png', 'bmp'])
            if uploaded_file:
                image = Image.open(uploaded_file)
        
        elif input_method == "📸 Camera":
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                image = Image.open(camera_image)
        
        elif input_method == "🔗 Image URL":
            url = st.text_input("Enter image URL")
            if url:
                try:
                    import requests
                    response = requests.get(url)
                    image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    st.error(f"Error loading image: {e}")
        
        if image is not None:
            # Process image
            with st.spinner("🔍 Analyzing image..."):
                annotated_image, detections, persons, vehicles = process_image(image, model)
            
            # Display results
            if annotated_image is not None:
                st.image(annotated_image, caption="Detection Results", use_container_width=True)
                
                # Person alert
                if persons:
                    st.markdown("""
                    <div class="person-alert">
                        ⚠️⚠️⚠️ PERSON DETECTED! ⚠️⚠️⚠️
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed alerts for each person
                    for person in persons:
                        alert_msg = f"⚠️ Person detected {person['direction']}, {person['distance']}!"
                        st.warning(alert_msg)
                        
                        # Browser speech
                        st.markdown(f"""
                        <script>
                            var msg = new SpeechSynthesisUtterance("{alert_msg}");
                            window.speechSynthesis.speak(msg);
                        </script>
                        """, unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.detection_history.append({
                            'timestamp': datetime.now(),
                            'type': 'person_alert',
                            'class': 'person',
                            'confidence': person['confidence'],
                            'direction': person['direction'],
                            'distance': person['distance']
                        })
                
                # Vehicle alerts
                for vehicle in vehicles:
                    if vehicle['distance'] in ['close', 'very close']:
                        alert_msg = f"⚠️ Vehicle {vehicle['direction']}, {vehicle['distance']}!"
                        st.warning(alert_msg)
                        
                        st.session_state.detection_history.append({
                            'timestamp': datetime.now(),
                            'type': 'vehicle_alert',
                            'class': vehicle['class'],
                            'confidence': vehicle['confidence'],
                            'direction': vehicle['direction'],
                            'distance': vehicle['distance']
                        })
                
                # Save all detections
                for detection in detections:
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'type': 'detection',
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'distance': detection['distance'],
                        'direction': detection['direction']
                    })
    
    with col2:
        st.markdown("### 📈 Analysis")
        
        if 'detections' in locals() and detections:
            # Summary stats
            person_count = len(persons)
            vehicle_count = len(vehicles)
            
            st.markdown(f"""
            <div class="stat-card">
                <h3>📊 Detection Summary</h3>
                <h1 style="font-size: 2rem;">👤 {person_count}</h1>
                <p>Persons Detected</p>
                <h2>🚗 {vehicle_count}</h2>
                <p>Vehicles Detected</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            if detections:
                avg_conf = np.mean([d['confidence'] for d in detections])
                st.plotly_chart(create_confidence_gauge(avg_conf), use_container_width=True)
            
            # Person details
            if persons:
                st.markdown("#### 👤 Person Details")
                for i, person in enumerate(persons):
                    color_class = "danger-card" if person['distance'] in ['close', 'very close'] else "detection-card"
                    st.markdown(f"""
                    <div class="{color_class}">
                        <strong>Person {i+1}</strong><br>
                        📍 Direction: {person['direction']}<br>
                        📏 Distance: {person['distance']}<br>
                        🎯 Confidence: {person['confidence']:.2%}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Vehicle details
            if vehicles:
                st.markdown("#### 🚗 Vehicle Details")
                for vehicle in vehicles:
                    color_class = "warning-card" if vehicle['distance'] in ['close', 'very close'] else "detection-card"
                    st.markdown(f"""
                    <div class="{color_class}">
                        <strong>{vehicle['class'].upper()}</strong><br>
                        📍 Direction: {vehicle['direction']}<br>
                        📏 Distance: {vehicle['distance']}<br>
                        🎯 Confidence: {vehicle['confidence']:.2%}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Detection History
    st.markdown("---")
    st.markdown("### 📜 Detection History")
    
    if st.session_state.detection_history:
        # Create dataframe
        df_history = pd.DataFrame(st.session_state.detection_history[-30:])
        
        if not df_history.empty:
            # Format for display
            display_cols = ['timestamp', 'class', 'direction', 'distance', 'confidence']
            if all(col in df_history.columns for col in display_cols):
                display_df = df_history[display_cols].copy()
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
                display_df = display_df.rename(columns={
                    'timestamp': 'Time',
                    'class': 'Object',
                    'direction': 'Direction',
                    'distance': 'Distance',
                    'confidence': 'Confidence'
                })
                
                st.dataframe(display_df, use_container_width=True, height=300)
        
        # Timeline chart
        timeline = create_timeline_chart()
        if timeline:
            st.plotly_chart(timeline, use_container_width=True)
        
        # Clear history
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.detection_history = []
                st.rerun()
    else:
        st.info("No detections yet. Upload an image or take a photo to start!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; opacity: 0.7;">
        <p>🦯 Smart Blind Stick System | AI-Powered Vision Assistance | Emergency Ready</p>
        <p>💡 Tip: For best results, ensure good lighting and clear images</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
