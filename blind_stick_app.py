import cv2
import numpy as np
import threading
import time
import queue
import warnings
import asyncio
import json
import websockets
import socket
import requests
import subprocess
import re
import serial
import serial.tools.list_ports
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import os
import platform
from ultralytics import YOLO
import pyttsx3
import math
import geocoder  # Add this for better location
from geopy.geocoders import Nominatim  # For reverse geocoding

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Your Google Maps API Key
GOOGLE_MAPS_API_KEY = "AIzaSyCdQGVYnjmSAzxnTu4g_zEXKGhgzqbZDvc"

# Bluetooth Configuration
BLUETOOTH_PORT = None
BLUETOOTH_BAUD = 9600
bluetooth_serial = None

# Voice alert tracking for person detection
last_person_alert_time = {}
person_alert_cooldown = 3  # seconds between repeated person alerts

def find_bluetooth_port():
    """Find the Bluetooth COM port automatically"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Bluetooth" in port.description or "HC-05" in port.description or "HC-06" in port.description:
            print(f"✅ Found Bluetooth device: {port.device} - {port.description}")
            return port.device
        elif "COM" in port.device and "USB" not in port.description:
            print(f"🔍 Testing port: {port.device} - {port.description}")
            try:
                test_ser = serial.Serial(port.device, 9600, timeout=1)
                test_ser.write(b"AT\r\n")
                time.sleep(0.5)
                response = test_ser.read(100)
                test_ser.close()
                if "OK" in response.decode('utf-8', errors='ignore'):
                    print(f"✅ Found Bluetooth module on: {port.device}")
                    return port.device
            except:
                pass
    return None

def init_bluetooth():
    """Initialize Bluetooth connection"""
    global bluetooth_serial, BLUETOOTH_PORT
    
    BLUETOOTH_PORT = find_bluetooth_port()
    
    if BLUETOOTH_PORT:
        try:
            bluetooth_serial = serial.Serial(BLUETOOTH_PORT, BLUETOOTH_BAUD, timeout=1)
            time.sleep(2)
            print(f"✅ Bluetooth connected on {BLUETOOTH_PORT}")
            bluetooth_serial.write(b"Python Connected\r\n")
            return True
        except Exception as e:
            print(f"⚠️ Bluetooth connection error: {e}")
            return False
    else:
        print("⚠️ No Bluetooth module found. Running without Bluetooth.")
        return False

def send_to_arduino(message):
    """Send command to Arduino via Bluetooth"""
    global bluetooth_serial
    if bluetooth_serial and bluetooth_serial.is_open:
        try:
            bluetooth_serial.write(f"{message}\n".encode())
            return True
        except Exception as e:
            print(f"Bluetooth send error: {e}")
            return False
    return False

# MongoDB Manager Class
class MongoDBManager:
    def __init__(self, connection_string=None):
        try:
            if connection_string is None:
                connection_string = "mongodb://localhost:27017/"
            
            self.client = MongoClient(connection_string)
            self.db = self.client['blind_stick_db']
            
            self.alerts = self.db['alerts']
            self.detections = self.db['detections']
            self.system_logs = self.db['system_logs']
            self.emergency_events = self.db['emergency_events']
            self.person_detections = self.db['person_detections']  # New collection for person tracking
            
            self.alerts.create_index([('timestamp', -1)])
            self.alerts.create_index([('alert_type', 1)])
            self.detections.create_index([('timestamp', -1)])
            self.detections.create_index([('object_type', 1)])
            self.system_logs.create_index([('timestamp', -1)])
            self.person_detections.create_index([('timestamp', -1)])
            
            print("✅ MongoDB connected successfully!")
            
        except Exception as e:
            print(f"⚠️ MongoDB connection error: {e}")
            self.client = None
    
    def save_person_detection(self, person_data):
        """Save person detection with location"""
        try:
            if self.client is None:
                return False
            
            detection_data = {
                'person_id': person_data.get('person_id', 0),
                'distance': person_data.get('distance', 'unknown'),
                'direction': person_data.get('direction', 'unknown'),
                'confidence': person_data.get('confidence', 0),
                'location': person_data.get('location', {}),
                'lat': person_data.get('location', {}).get('lat', 0),
                'lng': person_data.get('location', {}).get('lng', 0),
                'timestamp': datetime.now(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'hour': datetime.now().hour
            }
            self.person_detections.insert_one(detection_data)
            return True
        except Exception as e:
            print(f"Error saving person detection: {e}")
            return False
    
    def save_alert(self, alert_type, message, location, person_count):
        try:
            if self.client is None:
                return False
            
            alert_data = {
                'alert_type': alert_type,
                'message': message,
                'location': location,
                'lat': location.get('lat', 0),
                'lng': location.get('lng', 0),
                'person_count': person_count,
                'timestamp': datetime.now(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'hour': datetime.now().hour
            }
            self.alerts.insert_one(alert_data)
            return True
        except Exception as e:
            print(f"Error saving alert: {e}")
            return False
    
    def save_detection(self, object_type, confidence, distance, direction):
        try:
            if self.client is None:
                return False
            
            detection_data = {
                'object_type': object_type,
                'confidence': confidence,
                'distance': distance,
                'direction': direction,
                'timestamp': datetime.now(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'hour': datetime.now().hour
            }
            self.detections.insert_one(detection_data)
            return True
        except Exception as e:
            print(f"Error saving detection: {e}")
            return False
    
    def save_emergency(self, location, person_count, detections):
        try:
            if self.client is None:
                return False
            
            emergency_data = {
                'location': location,
                'lat': location.get('lat', 0),
                'lng': location.get('lng', 0),
                'person_count': person_count,
                'detected_objects': detections,
                'timestamp': datetime.now(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'resolved': False
            }
            self.emergency_events.insert_one(emergency_data)
            return True
        except Exception as e:
            print(f"Error saving emergency: {e}")
            return False
    
    def get_recent_detections(self, limit=50):
        try:
            if self.client is None:
                return []
            detections = list(self.detections.find({}, {'_id': 0}).sort('timestamp', -1).limit(limit))
            return detections
        except Exception as e:
            print(f"Error getting detections: {e}")
            return []
    
    def get_recent_alerts(self, limit=50):
        try:
            if self.client is None:
                return []
            alerts = list(self.alerts.find({}, {'_id': 0}).sort('timestamp', -1).limit(limit))
            return alerts
        except Exception as e:
            print(f"Error getting alerts: {e}")
            return []
    
    def get_stats(self):
        try:
            if self.client is None:
                return {}
            today = datetime.now().strftime('%Y-%m-%d')
            stats = {
                'total_detections': self.detections.count_documents({}),
                'total_alerts': self.alerts.count_documents({}),
                'today_detections': self.detections.count_documents({'date': today}),
                'today_alerts': self.alerts.count_documents({'date': today}),
                'emergency_events': self.emergency_events.count_documents({}),
                'active_emergencies': self.emergency_events.count_documents({'resolved': False}),
                'person_detections_today': self.person_detections.count_documents({'date': today})
            }
            return stats
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def get_detections_by_type(self):
        try:
            if self.client is None:
                return []
            results = list(self.detections.aggregate([
                {'$group': {
                    '_id': '$object_type',
                    'count': {'$sum': 1},
                    'avg_confidence': {'$avg': '$confidence'}
                }},
                {'$sort': {'count': -1}}
            ]))
            return results
        except Exception as e:
            print(f"Error getting detection types: {e}")
            return []
    
    def get_alerts_by_type(self):
        try:
            if self.client is None:
                return []
            results = list(self.alerts.aggregate([
                {'$group': {
                    '_id': '$alert_type',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}}
            ]))
            return results
        except Exception as e:
            print(f"Error getting alert types: {e}")
            return []
    
    def log_system_event(self, event_type, details):
        try:
            if self.client is None:
                return False
            log_data = {
                'event_type': event_type,
                'details': details,
                'timestamp': datetime.now()
            }
            self.system_logs.insert_one(log_data)
            return True
        except Exception as e:
            print(f"Error logging event: {e}")
            return False
    
    def close(self):
        if self.client:
            self.client.close()

class SmartBlindStick:
    def __init__(self, mongodb_uri=None):
        self.db = MongoDBManager(mongodb_uri)
        
        # Initialize text-to-speech
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.tts_available = True
            print("✅ Text-to-speech initialized")
        except:
            print("⚠️ Text-to-speech not available")
            self.tts_available = False
        
        self.speech_queue = queue.Queue()
        self.last_spoken = {}
        self.speech_cooldown = {
            'person': 2.5,  # Reduced for faster person alerts
            'person_direction': 2.0,  # Specific cooldown for direction alerts
            'stairs': 3.0, 
            'pothole': 2.5, 
            'wall': 2.5, 
            'vehicle': 2.0, 
            'emergency': 10.0
        }
        
        # Track last person alert time for each direction
        self.last_person_alert_time = {}
        self.person_alert_cooldown = 2.5  # seconds between person alerts
        
        # Load YOLO model
        print("Loading YOLO model...")
        try:
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded!")
        except Exception as e:
            print(f"⚠️ YOLO not available: {e}")
            self.model = None
        
        # Important classes for detection
        self.important_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 11: 'stop sign', 13: 'bench', 56: 'chair', 57: 'couch',
            59: 'bed', 61: 'toilet', 62: 'tv', 63: 'laptop', 67: 'cell phone'
        }
        
        # Camera setup
        self.cap = None
        print("Opening camera...")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap = cap
                print(f"✅ Camera {i} opened successfully!")
                break
            else:
                cap.release()
        
        if self.cap is None:
            print("❌ No camera found! Using test pattern.")
        
        self.clients = set()
        self.current_data = {}
        self.emergency_mode = False
        self.person_count = 0
        self.vehicle_count = 0
        self.detected_objects = []
        self.fps = 0
        self.detection_count = 0
        self.location_accuracy = "unknown"
        self.mobile_location = None  # Store mobile GPS location
        self.last_person_location = None  # Store last person detection location
        
        # Get initial location with enhanced accuracy
        self.current_location = self.get_enhanced_laptop_location()
        
        # Alert tracking for Bluetooth
        self.last_bluetooth_alert_time = 0
        self.bluetooth_alert_cooldown = 1.0  # seconds
        
        # Start threads
        threading.Thread(target=self.process_speech_queue, daemon=True).start()
        threading.Thread(target=self.update_location_enhanced, daemon=True).start()
        threading.Thread(target=self.read_arduino_data, daemon=True).start()
        
        self.db.log_system_event('SYSTEM_START', 'Smart Blind Stick system initialized')
    
    def read_arduino_data(self):
        """Read sensor data from Arduino via Bluetooth"""
        global bluetooth_serial
        
        while True:
            if bluetooth_serial and bluetooth_serial.is_open:
                try:
                    if bluetooth_serial.in_waiting:
                        line = bluetooth_serial.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            print(f"📟 Arduino: {line}")
                            
                            if line.startswith("SENSOR_DATA:"):
                                data = line[12:]
                                parts = data.split(',')
                                sensor_data = {}
                                for part in parts:
                                    if '=' in part:
                                        key, val = part.split('=')
                                        sensor_data[key] = val
                                
                                self.sensor_data = sensor_data
                                
                            elif line.startswith("SENSOR_ALERT:"):
                                alert_data = line[13:]
                                print(f"⚠️ Arduino Alert: {alert_data}")
                                self.db.save_alert('sensor', alert_data, self.current_location, self.person_count)
                                
                except Exception as e:
                    print(f"Bluetooth read error: {e}")
            time.sleep(0.1)
    
    def send_alert_to_arduino(self, alert_type, details=""):
        """Send alert to Arduino to trigger buzzer"""
        global bluetooth_serial
        
        current_time = time.time()
        if current_time - self.last_bluetooth_alert_time < self.bluetooth_alert_cooldown:
            return
        
        self.last_bluetooth_alert_time = current_time
        
        message = f"CAMERA_ALERT:{alert_type}"
        if details:
            message += f",{details}"
        
        send_to_arduino(message)
        print(f"📤 Sent to Arduino: {message}")
    
    def clear_arduino_alert(self):
        """Clear active alert on Arduino"""
        send_to_arduino("CAMERA_CLEAR")
    
    def get_phone_gps_location(self):
        """Get GPS location from connected phone via WebSocket"""
        # This will be updated when phone sends location
        return self.mobile_location
    
    def get_wifi_location_windows(self):
        """Get location from WiFi networks using Google Geolocation API"""
        try:
            result = subprocess.run(['netsh', 'wlan', 'show', 'networks', 'mode=bssid'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                ssid_match = re.findall(r'SSID\s+:\s(.+)', result.stdout)
                signal_match = re.findall(r'Signal\s+:\s(\d+)%', result.stdout)
                mac_match = re.findall(r'BSSID\s+\d+\s+:\s([0-9A-Fa-f:-]+)', result.stdout)
                
                if ssid_match and signal_match:
                    wifi_data = {
                        "considerIp": "true",
                        "wifiAccessPoints": []
                    }
                    
                    for i in range(min(len(ssid_match), 5)):
                        wifi_data["wifiAccessPoints"].append({
                            "macAddress": mac_match[i] if i < len(mac_match) else "00:00:00:00:00:00",
                            "signalStrength": int(signal_match[i]) if i < len(signal_match) else -60,
                            "age": 0,
                            "channel": 6,
                            "signalToNoiseRatio": 0
                        })
                    
                    try:
                        response = requests.post(
                            f'https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_MAPS_API_KEY}',
                            json=wifi_data,
                            timeout=5
                        )
                        if response.status_code == 200:
                            data = response.json()
                            if 'location' in data:
                                lat = data['location']['lat']
                                lng = data['location']['lng']
                                accuracy = data.get('accuracy', 100)
                                return {
                                    "lat": lat,
                                    "lng": lng,
                                    "address": f"WiFi Location (accuracy: {accuracy}m)",
                                    "accuracy": "wifi",
                                    "source": "wifi_geolocation"
                                }
                    except Exception as e:
                        print(f"Google Geolocation API error: {e}")
            
            return None
        except Exception as e:
            print(f"WiFi location error: {e}")
            return None
    
    def get_ip_location_enhanced(self):
        """Enhanced IP geolocation with multiple services"""
        services = [
            {
                'url': 'http://ip-api.com/json/',
                'parser': lambda d: {
                    'lat': d.get('lat'),
                    'lng': d.get('lon'),
                    'city': d.get('city'),
                    'region': d.get('regionName'),
                    'country': d.get('country'),
                    'accuracy': 'city'
                }
            },
            {
                'url': 'https://ipapi.co/json/',
                'parser': lambda d: {
                    'lat': d.get('latitude'),
                    'lng': d.get('longitude'),
                    'city': d.get('city'),
                    'region': d.get('region'),
                    'country': d.get('country_name'),
                    'accuracy': 'city'
                }
            }
        ]
        
        for service in services:
            try:
                response = requests.get(service['url'], timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    parsed = service['parser'](data)
                    if parsed['lat'] and parsed['lng']:
                        return {
                            "lat": parsed['lat'],
                            "lng": parsed['lng'],
                            "address": f"{parsed['city']}, {parsed['region']}, {parsed['country']}",
                            "city": parsed['city'],
                            "region": parsed['region'],
                            "country": parsed['country'],
                            "accuracy": parsed['accuracy'],
                            "source": "ip_geolocation"
                        }
            except:
                continue
        return None
    
    def get_manual_perundurai_location(self):
        """Manual location for Perundurai"""
        return {
            "lat": 11.2745,
            "lng": 77.5831,
            "address": "Perundurai, Erode District, Tamil Nadu 638052",
            "city": "Perundurai",
            "region": "Tamil Nadu",
            "country": "India",
            "accuracy": "manual",
            "source": "manual_set",
            "description": "Perundurai Town Center"
        }
    
    def get_enhanced_laptop_location(self):
        """Get the most accurate laptop location using multiple methods"""
        
        print("\n🔍 Searching for laptop location...")
        
        # Method 1: Phone GPS (most accurate)
        phone_location = self.get_phone_gps_location()
        if phone_location:
            print(f"✅ Phone GPS Location: {phone_location.get('address', 'GPS location')}")
            print(f"📡 Coordinates: {phone_location['lat']:.6f}, {phone_location['lng']:.6f}")
            self.location_accuracy = "high"
            return phone_location
        
        # Method 2: WiFi Positioning
        wifi_location = self.get_wifi_location_windows()
        if wifi_location:
            print(f"✅ WiFi Location: {wifi_location['address']}")
            print(f"📡 Coordinates: {wifi_location['lat']:.6f}, {wifi_location['lng']:.6f}")
            self.location_accuracy = "high"
            return wifi_location
        
        # Method 3: Enhanced IP Geolocation
        ip_location = self.get_ip_location_enhanced()
        if ip_location:
            print(f"✅ IP Location: {ip_location['address']}")
            print(f"📍 Coordinates: {ip_location['lat']:.6f}, {ip_location['lng']:.6f}")
            self.location_accuracy = "medium"
            return ip_location
        
        # Method 4: Manual Perundurai Location
        print("📍 Using Manual Perundurai Location")
        perundurai_location = self.get_manual_perundurai_location()
        print(f"📍 Location: {perundurai_location['address']}")
        print(f"📌 Coordinates: {perundurai_location['lat']:.6f}, {perundurai_location['lng']:.6f}")
        self.location_accuracy = "manual"
        return perundurai_location
    
    def update_location_enhanced(self):
        """Update location every 15 seconds (more frequent for accuracy)"""
        last_update = 0
        while True:
            try:
                current_time = time.time()
                if current_time - last_update > 15:
                    new_location = self.get_enhanced_laptop_location()
                    if new_location and new_location.get('lat'):
                        self.current_location = new_location
                        print(f"📍 Location updated: {self.current_location['address']}")
                        self.db.log_system_event('LOCATION_UPDATE', 
                            f"Location: {new_location['address']} ({new_location.get('source', 'unknown')})")
                        last_update = current_time
            except Exception as e:
                print(f"Location update error: {e}")
            time.sleep(5)
    
    def process_speech_queue(self):
        """Process speech queue"""
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                if self.tts_available:
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    print(f"🔊 VOICE: {text}")
                self.speech_queue.task_done()
            except:
                pass
    
    def speak(self, text, alert_type='obstacle'):
        """Speak with cooldown"""
        now = time.time()
        if alert_type in self.last_spoken:
            if now - self.last_spoken[alert_type] < self.speech_cooldown.get(alert_type, 2):
                return
        self.last_spoken[alert_type] = now
        self.speech_queue.put(text)
        
        if alert_type != 'system':
            self.db.save_alert(alert_type, text, self.current_location, self.person_count)
        
        print(f"🔊 Speaking: {text}")
    
    def speak_person_alert(self, direction, distance, person_id=0):
        """Enhanced person alert with direction and distance"""
        # Create a unique key for cooldown based on direction
        alert_key = f"person_{direction}_{distance}"
        
        now = time.time()
        if alert_key in self.last_person_alert_time:
            if now - self.last_person_alert_time[alert_key] < self.person_alert_cooldown:
                return
        
        self.last_person_alert_time[alert_key] = now
        
        # Create detailed alert message
        if distance == "very close":
            message = f"Warning! Person {direction}, very close to you!"
        elif distance == "close":
            message = f"Person detected {direction}"
        else:
            message = f"There is a person {direction}"
        
        # Add location context if available
        if self.current_location and self.current_location.get('address'):
            location_short = self.current_location['address'].split(',')[0]
            message += f" at {location_short}"
        
        # Speak the alert
        self.speak(message, 'person')
        
        # Send to Arduino for buzzer
        if distance in ["very close", "close"]:
            self.send_alert_to_arduino(f"PERSON_{distance.upper()}", direction)
        
        # Save to database with location
        person_data = {
            'person_id': person_id,
            'distance': distance,
            'direction': direction,
            'confidence': 0.8,  # Default confidence
            'location': self.current_location
        }
        self.db.save_person_detection(person_data)
        
        # Store last person location for tracking
        self.last_person_location = self.current_location.copy()
        self.last_person_location['person_direction'] = direction
        self.last_person_location['person_distance'] = distance
    
    def detect_with_yolo(self, frame):
        """Detect objects using YOLO with enhanced person alerts"""
        detections = []
        height, width = frame.shape[:2]
        
        if self.model is None:
            return frame, detections
        
        results = self.model(frame, stream=True, conf=0.5)
        
        person_id_counter = 0
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf < 0.5:
                        continue
                    
                    class_name = self.important_classes.get(cls, f"object_{cls}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    box_height = y2 - y1
                    if box_height > height * 0.5:
                        distance = "very close"
                    elif box_height > height * 0.3:
                        distance = "close"
                    elif box_height > height * 0.15:
                        distance = "medium"
                    else:
                        distance = "far"
                    
                    center_x = (x1 + x2) / 2
                    if center_x < width * 0.3:
                        direction = "on your left"
                    elif center_x > width * 0.7:
                        direction = "on your right"
                    else:
                        direction = "straight ahead"
                    
                    detection = {
                        'class': class_name,
                        'confidence': conf,
                        'distance': distance,
                        'direction': direction,
                        'bbox': (x1, y1, x2, y2)
                    }
                    detections.append(detection)
                    
                    # Determine color based on distance
                    if distance == "very close":
                        color = (0, 0, 255)
                    elif distance == "close":
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {conf:.2f} ({distance}, {direction})"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # ENHANCED PERSON DETECTION WITH VOICE ALERT
                    if class_name == 'person':
                        # Speak detailed person alert
                        self.speak_person_alert(direction, distance, person_id_counter)
                        
                        # Add extra visual indicator for persons
                        cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 20, (0, 0, 255), 2)
                        cv2.putText(frame, "⚠️ PERSON ⚠️", (x1, y2+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Send to Arduino for immediate buzzer
                        if distance == "very close":
                            self.send_alert_to_arduino("PERSON_VERY_CLOSE_ALERT", direction)
                        elif distance == "close":
                            self.send_alert_to_arduino("PERSON_CLOSE_ALERT", direction)
                        
                        self.db.save_detection('person', conf, distance, direction)
                        self.detection_count += 1
                        person_id_counter += 1
                    
                    elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
                        if distance == "close" or distance == "very close":
                            self.speak(f"Vehicle {direction}, {distance}!", 'vehicle')
                            self.send_alert_to_arduino(f"VEHICLE_{distance.upper()}", direction)
                        self.db.save_detection('vehicle', conf, distance, direction)
                        self.detection_count += 1
        
        return frame, detections
    
    def detect_walls_advanced(self, frame):
        """Wall detection"""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        wall_detected = False
        wall_distance = "far"
        wall_direction = "ahead"
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=50)
        
        if lines is not None:
            vertical_lines = []
            line_positions = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                if 70 < angle < 110:
                    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if line_length > height * 0.3:
                        vertical_lines.append(line)
                        line_positions.append(x1)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            if len(vertical_lines) > 5:
                wall_detected = True
                density = len(vertical_lines) / (width * height) * 10000
                
                if density > 1.5:
                    wall_distance = "very close"
                elif density > 0.8:
                    wall_distance = "close"
                elif density > 0.3:
                    wall_distance = "medium"
                
                avg_position = sum(line_positions) / len(line_positions)
                if avg_position < width * 0.3:
                    wall_direction = "left"
                elif avg_position > width * 0.7:
                    wall_direction = "right"
        
        if wall_detected:
            color = (0, 0, 255) if wall_distance == "very close" else (0, 165, 255)
            cv2.putText(frame, f"WALL: {wall_distance}, {wall_direction}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if wall_distance == "very close":
                self.speak(f"Wall {wall_direction}, very close!", 'wall')
                self.send_alert_to_arduino(f"WALL_VERY_CLOSE", wall_direction)
            elif wall_distance == "close":
                self.speak(f"Wall {wall_direction}", 'wall')
                self.send_alert_to_arduino(f"WALL_CLOSE", wall_direction)
        
        return frame, wall_detected, wall_direction, wall_distance
    
    def detect_stairs_advanced(self, frame):
        """Stairs detection"""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        stairs_detected = False
        stairs_conf = 0
        stairs_location = None
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=30)
        
        if lines is not None and len(lines) > 5:
            horiz_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                if angle < 20 or angle > 160:
                    horiz_lines.append((y1 + y2) // 2)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if len(horiz_lines) > 3:
                horiz_lines.sort()
                gaps = [horiz_lines[i+1] - horiz_lines[i] for i in range(len(horiz_lines)-1)]
                
                if gaps and len(gaps) > 0:
                    avg_gap = sum(gaps) / len(gaps)
                    consistency = sum(abs(g - avg_gap) for g in gaps) / len(gaps)
                    
                    if consistency < 25 and 20 < avg_gap < 80:
                        stairs_conf = min(95, 50 + len(horiz_lines) * 8)
                        stairs_detected = True
                        
                        if horiz_lines:
                            stairs_y = sum(horiz_lines) / len(horiz_lines)
                            if stairs_y < height * 0.4:
                                stairs_location = "ahead"
                            elif stairs_y < height * 0.7:
                                stairs_location = "in front"
                            else:
                                stairs_location = "below"
        
        if stairs_detected:
            color = (0, 0, 255) if stairs_conf > 70 else (0, 165, 255)
            cv2.putText(frame, f"STAIRS DETECTED! ({stairs_conf:.0f}%)", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if stairs_location:
                cv2.putText(frame, f"Location: {stairs_location}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if stairs_conf > 70:
                self.speak(f"Stairs detected {stairs_location}!", 'stairs')
                self.send_alert_to_arduino("STAIRS_DETECTED", stairs_location)
        
        return frame, stairs_detected, stairs_conf, stairs_location
    
    def detect_potholes_advanced(self, frame):
        """Pothole detection"""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        pothole_detected = False
        pothole_conf = 0
        pothole_location = None
        
        lower_half = gray[height//2:, :]
        
        blurred = cv2.GaussianBlur(lower_half, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if 400 < area < 8000:
                peri = cv2.arcLength(cnt, True)
                if peri > 0:
                    circularity = 4 * np.pi * area / (peri * peri)
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / h
                    
                    if circularity > 0.4 and 0.5 < aspect_ratio < 2.0:
                        pothole_conf = min(90, 50 + area/100)
                        pothole_detected = True
                        
                        center_x = x + w/2
                        if center_x < width * 0.3:
                            pothole_location = "left"
                        elif center_x > width * 0.7:
                            pothole_location = "right"
                        else:
                            pothole_location = "center"
                        
                        y_global = y + height//2
                        cv2.rectangle(frame, (x, y_global), (x+w, y_global+h), (0, 0, 255), 2)
                        cv2.putText(frame, f"POTHOLE! ({pothole_conf:.0f}%)", 
                                   (x, y_global-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        if pothole_conf > 65:
                            self.speak(f"Pothole detected {pothole_location}!", 'pothole')
                            self.send_alert_to_arduino("POTHOLE_DETECTED", pothole_location)
        
        if pothole_detected:
            cv2.putText(frame, f"⚠️ POTHOLE WARNING! ({pothole_conf:.0f}%)", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame, pothole_detected, pothole_conf, pothole_location
    
    def generate_frames(self):
        """Generate video frames"""
        fps_start = time.time()
        frame_count = 0
        last_clear_time = time.time()
        
        while True:
            if self.cap is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Not Available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Please check camera connection", (130, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                detections = []
                wall_detected = False
                stairs_detected = False
                pothole_detected = False
                self.fps = 30
                
            else:
                ret, frame = self.cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera Error", (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    frame_count += 1
                    if frame_count % 30 == 0:
                        fps = 30 / (time.time() - fps_start) if (time.time() - fps_start) > 0 else 30
                        fps_start = time.time()
                        self.fps = int(fps)
                    
                    frame, yolo_detections = self.detect_with_yolo(frame)
                    frame, wall_detected, wall_dir, wall_dist = self.detect_walls_advanced(frame)
                    frame, stairs_detected, stairs_conf, stairs_loc = self.detect_stairs_advanced(frame)
                    frame, pothole_detected, pothole_conf, pothole_loc = self.detect_potholes_advanced(frame)
                    
                    self.person_count = sum(1 for d in yolo_detections if d['class'] == 'person')
                    self.vehicle_count = sum(1 for d in yolo_detections if d['class'] in ['car', 'truck', 'bus', 'motorcycle'])
                    self.detected_objects = yolo_detections
                    
                    # Clear Arduino alert if no detections for 2 seconds
                    if not (wall_detected or stairs_detected or pothole_detected or self.person_count > 0):
                        if time.time() - last_clear_time > 2:
                            self.clear_arduino_alert()
                            last_clear_time = time.time()
            
            # Display information on frame
            cv2.putText(frame, "SMART BLIND STICK SYSTEM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self.fps} | Persons: {self.person_count} | Vehicles: {self.vehicle_count}", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"Mobile Connected: {len(self.clients)} | Bluetooth: {'✓' if bluetooth_serial else '✗'}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Show last person detection location
            if self.last_person_location:
                cv2.putText(frame, f"Last Person: {self.last_person_location.get('person_direction', 'unknown')}", 
                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            loc_source = self.current_location.get('source', 'unknown')
            source_icon = "📡" if loc_source == "wifi_geolocation" else "🌐" if loc_source == "ip_geolocation" else "📍"
            cv2.putText(frame, f"{source_icon} Source: {loc_source}", (10, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            
            loc_text = f"Location: {self.current_location.get('address', 'Unknown')[:35]}"
            cv2.putText(frame, loc_text, (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            coord_text = f"Coords: {self.current_location.get('lat', 0):.6f}, {self.current_location.get('lng', 0):.6f}"
            cv2.putText(frame, coord_text, (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            if self.emergency_mode:
                cv2.putText(frame, "EMERGENCY MODE ACTIVE", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            self.current_data = {
                'detections': yolo_detections if 'yolo_detections' in locals() else [],
                'person_count': self.person_count,
                'vehicle_count': self.vehicle_count,
                'walls': {'detected': wall_detected, 'direction': wall_dir, 'distance': wall_dist},
                'stairs': {'detected': stairs_detected, 'confidence': stairs_conf, 'location': stairs_loc},
                'potholes': {'detected': pothole_detected, 'confidence': pothole_conf, 'location': pothole_loc},
                'fps': self.fps,
                'emergency': self.emergency_mode,
                'location': self.current_location,
                'last_person_location': self.last_person_location,
                'timestamp': datetime.now().isoformat(),
                'connected_clients': len(self.clients),
                'detection_count': self.detection_count,
                'location_source': self.current_location.get('source', 'unknown'),
                'bluetooth_connected': bluetooth_serial is not None and bluetooth_serial.is_open
            }
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    async def handle_client(self, websocket):
        """Handle WebSocket clients with GPS location support"""
        self.clients.add(websocket)
        print(f"📱 Mobile connected! Total: {len(self.clients)}")
        self.db.log_system_event('CLIENT_CONNECTED', f"Total clients: {len(self.clients)}")
        
        try:
            if self.current_data:
                await websocket.send(json.dumps(self.current_data))
            
            async for message in websocket:
                data = json.loads(message)
                if data.get('type') == 'register':
                    print("✅ Mobile registered")
                    await websocket.send(json.dumps({'type': 'registered', 'status': 'ok'}))
                
                elif data.get('type') == 'gps_location':
                    # Receive GPS location from mobile phone
                    gps_data = data.get('location', {})
                    if gps_data.get('lat') and gps_data.get('lng'):
                        self.mobile_location = {
                            'lat': gps_data['lat'],
                            'lng': gps_data['lng'],
                            'address': f"GPS: {gps_data['lat']:.6f}, {gps_data['lng']:.6f}",
                            'accuracy': gps_data.get('accuracy', 10),
                            'source': 'phone_gps',
                            'timestamp': datetime.now().isoformat()
                        }
                        # Update current location with GPS
                        self.current_location = self.mobile_location
                        print(f"📍 Phone GPS Location: {self.mobile_location['lat']:.6f}, {self.mobile_location['lng']:.6f}")
                        
                        # Send acknowledgment
                        await websocket.send(json.dumps({
                            'type': 'gps_received',
                            'status': 'ok',
                            'location': self.mobile_location
                        }))
                
                elif data.get('type') == 'request_location':
                    await websocket.send(json.dumps({
                        'type': 'location_update',
                        'location': self.current_location
                    }))
                
                elif data.get('type') == 'test_buzzer':
                    send_to_arduino("TEST_BUZZER")
                    await websocket.send(json.dumps({'type': 'buzzer_test', 'status': 'sent'}))
                
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.clients.remove(websocket)
            print(f"📱 Mobile disconnected. Total: {len(self.clients)}")
            self.db.log_system_event('CLIENT_DISCONNECTED', f"Total clients: {len(self.clients)}")
    
    async def broadcast_updates(self):
        """Broadcast updates to all clients"""
        while True:
            if self.clients and self.current_data:
                self.current_data['connected_clients'] = len(self.clients)
                
                dead = set()
                for client in self.clients:
                    try:
                        await client.send(json.dumps(self.current_data))
                    except:
                        dead.add(client)
                self.clients -= dead
            await asyncio.sleep(0.1)
    
    def run_websocket(self):
        """Run WebSocket server"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.ws_loop = loop
        
        async def server():
            async with websockets.serve(self.handle_client, '0.0.0.0', 8765):
                print("🔌 WebSocket server running on ws://0.0.0.0:8765")
                await asyncio.gather(self.broadcast_updates(), asyncio.Future())
        
        loop.run_until_complete(server())
    
    async def send_emergency(self, location, person_count):
        """Send emergency alert with location"""
        maps_url = f"https://www.google.com/maps?q={location['lat']},{location['lng']}"
        
        # Send emergency to Arduino
        send_to_arduino("EMERGENCY")
        
        emergency_data = {
            'type': 'emergency',
            'title': '🚨 EMERGENCY ALERT! 🚨',
            'message': f'Emergency button pressed! Immediate assistance needed at {location["address"]}',
            'location': location,
            'maps_url': maps_url,
            'person_count': person_count,
            'vehicle_count': self.vehicle_count,
            'detected_objects': self.detected_objects[:5],
            'last_person_location': self.last_person_location,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'lat': location['lat'],
            'lng': location['lng'],
            'location_source': location.get('source', 'unknown')
        }
        
        print(f"\n{'='*60}")
        print("🚨 EMERGENCY ALERT SENT!")
        print(f"{'='*60}")
        print(f"📍 Location: {location['address']}")
        print(f"📍 Coordinates: {location['lat']}, {location['lng']}")
        print(f"📍 Source: {location.get('source', 'unknown')}")
        print(f"📍 Google Maps: {maps_url}")
        print(f"👥 Persons detected: {person_count}")
        print(f"🚗 Vehicles detected: {self.vehicle_count}")
        print(f"📱 Sending to {len(self.clients)} mobile device(s)...")
        print(f"🔊 Buzzer activated on Arduino")
        
        self.db.save_emergency(location, person_count, self.detected_objects)
        self.db.save_alert('emergency', f'Emergency alert at {location["address"]}', location, person_count)
        
        if self.clients:
            for client in self.clients:
                try:
                    await client.send(json.dumps(emergency_data))
                    print("✅ Alert sent to mobile")
                except Exception as e:
                    print(f"❌ Failed: {e}")
        else:
            print("⚠️ No mobile devices connected!")
            print("📱 Open http://[YOUR-IP]:5000 on your phone")
        
        print(f"{'='*60}\n")
        return emergency_data
    
    def run(self):
        """Start the system"""
        ws_thread = threading.Thread(target=self.run_websocket, daemon=True)
        ws_thread.start()
        
        self.speak("Smart Blind Stick system started. Person detection active.", "system")
        
        print("\n" + "="*60)
        print("🦯 SMART BLIND STICK SYSTEM WITH ENHANCED FEATURES")
        print("="*60)
        print("✅ System running!")
        print("✅ Person Detection with Voice Alerts - ACTIVE")
        print("✅ Direction & Distance Tracking - ACTIVE")
        print(f"✅ Bluetooth: {'Connected ✓' if bluetooth_serial else 'Not Connected ✗'}")
        print("✅ Database: MongoDB (blind_stick_db)")
        print("✅ WebSocket: port 8765")
        print("✅ Flask: port 5000")
        print(f"✅ Current Location: {self.current_location['address']}")
        print(f"✅ Coordinates: {self.current_location['lat']}, {self.current_location['lng']}")
        print(f"✅ Location Source: {self.current_location.get('source', 'unknown')}")
        print("\n📱 CONNECT YOUR MOBILE:")
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"   http://{local_ip}:5000")
        print("\n📍 ENHANCED LOCATION FEATURES:")
        print("   • Phone GPS will auto-update when mobile connects")
        print("   • Person detection includes location tracking")
        print("   • Emergency alerts include GPS coordinates")
        print("\n🎯 PERSON DETECTION VOICE ALERTS:")
        print("   • 'Warning! Person on your left, very close!'")
        print("   • 'Person detected straight ahead'")
        print("   • 'There is a person on your right'")
        print("\n🎯 GOOGLE MAPS FEATURES:")
        print("   • Interactive Google Maps with your location")
        print("   • Street View available")
        print("   • Satellite and terrain views")
        print("   • Directions and navigation")
        print("\n💡 Press 'E' key for emergency alert")
        print("="*60 + "\n")

# HTML Template with GPS Location Support
HTML_TEMPLATE = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>Smart Blind Stick - GPS Location Tracking</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 16px;
            color: #fff;
        }}
        .container {{ max-width: 500px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .header h1 {{ font-size: 24px; background: linear-gradient(135deg, #fff, #a8c0ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .header p {{ font-size: 12px; opacity: 0.7; margin-top: 5px; }}
        .status-badge {{ display: inline-block; margin-top: 8px; padding: 4px 12px; border-radius: 20px; font-size: 12px; }}
        .connected {{ background: rgba(76,175,80,0.3); border: 1px solid #4caf50; color: #4caf50; }}
        .disconnected {{ background: rgba(244,67,54,0.3); border: 1px solid #f44336; color: #f44336; }}
        .video-container {{ background: #000; border-radius: 16px; overflow: hidden; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
        .video-container img {{ width: 100%; display: block; }}
        .emergency-btn {{
            background: linear-gradient(135deg, #ff4444, #cc0000);
            border: none;
            width: 100%;
            padding: 16px;
            border-radius: 50px;
            color: white;
            font-size: 20px;
            font-weight: bold;
            cursor: pointer;
            margin-bottom: 16px;
            animation: pulse 2s infinite;
            box-shadow: 0 4px 12px rgba(255,0,0,0.3);
        }}
        @keyframes pulse {{ 0%,100%{{transform:scale(1);}} 50%{{transform:scale(1.02);}} }}
        .card {{ background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 16px; padding: 12px; margin-bottom: 12px; }}
        .stats-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px; }}
        .stat-box {{ background: rgba(0,0,0,0.4); padding: 8px; border-radius: 10px; text-align: center; }}
        .stat-label {{ font-size: 11px; opacity: 0.7; }}
        .stat-value {{ font-size: 20px; font-weight: bold; color: #4caf50; }}
        .person-alert {{
            background: rgba(255,68,68,0.3);
            border: 2px solid #ff4444;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 12px;
            text-align: center;
            animation: blink 1s infinite;
        }}
        @keyframes blink {{
            0%,100%{{background:rgba(255,68,68,0.3);}}
            50%{{background:rgba(255,68,68,0.6);}}
        }}
        .hazard-row {{ display: flex; gap: 8px; margin-top: 12px; }}
        .hazard-card {{ background: rgba(0,0,0,0.4); border-radius: 12px; padding: 10px; text-align: center; flex: 1; transition: all 0.3s; }}
        .hazard-card.active {{ background: rgba(255,68,68,0.3); border: 1px solid #ff4444; }}
        .detection-list {{ max-height: 250px; overflow-y: auto; }}
        .detection-item {{ background: rgba(255,255,255,0.1); padding: 10px; margin: 5px 0; border-radius: 10px; display: flex; justify-content: space-between; align-items: center; }}
        .detection-item.danger {{ background: rgba(255,68,68,0.3); border-left: 3px solid #ff4444; }}
        .detection-item.warning {{ background: rgba(255,152,0,0.2); border-left: 3px solid #ff9800; }}
        .alert-list {{ max-height: 200px; overflow-y: auto; }}
        .alert-item {{ background: rgba(0,0,0,0.4); padding: 10px; margin: 5px 0; border-radius: 10px; border-left: 3px solid #ff9800; }}
        .alert-item.emergency {{ border-left-color: #ff4444; background: rgba(255,68,68,0.2); animation: blinkAlert 0.5s; }}
        @keyframes blinkAlert {{ 0%,100%{{background:rgba(255,68,68,0.2);}} 50%{{background:rgba(255,68,68,0.4);}} }}
        .alert-time {{ font-size: 10px; opacity: 0.5; margin-bottom: 5px; }}
        .alert-title {{ font-weight: bold; font-size: 12px; margin-bottom: 3px; }}
        .alert-message {{ font-size: 11px; opacity: 0.8; }}
        #map {{ height: 350px; border-radius: 12px; margin-top: 8px; }}
        .location-text {{ font-size: 11px; text-align: center; margin-top: 8px; font-family: monospace; cursor: pointer; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 8px; }}
        .coordinates {{ font-size: 10px; color: #4caf50; margin-top: 5px; text-align: center; cursor: pointer; }}
        .source-badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 10px; margin-top: 5px; background: rgba(0,0,0,0.5); text-align: center; width: 100%; }}
        .map-buttons {{ display: flex; gap: 8px; margin-top: 8px; }}
        .map-btn {{ flex: 1; background: rgba(0,0,0,0.5); border: 1px solid #4caf50; padding: 8px; border-radius: 8px; color: white; cursor: pointer; text-align: center; font-size: 12px; transition: all 0.3s; }}
        .map-btn:hover {{ background: rgba(76,175,80,0.3); }}
        .map-type-selector {{ display: flex; gap: 5px; margin-top: 8px; }}
        .map-type-btn {{ flex: 1; background: rgba(0,0,0,0.5); border: none; padding: 6px; border-radius: 6px; color: white; cursor: pointer; font-size: 10px; }}
        .map-type-btn.active {{ background: #4caf50; }}
        .test-btn {{ background: rgba(33,150,243,0.3); border: 1px solid #2196f3; color: #2196f3; margin-top: 8px; }}
        .gps-status {{ font-size: 10px; text-align: center; margin-top: 5px; color: #4caf50; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦯 Smart Blind Stick <span style="font-size: 12px;">GPS Tracking</span></h1>
            <p>Real-time person detection & GPS location tracking</p>
            <div id="connectionStatus" class="status-badge connected">🟢 Connected</div>
            <div id="bluetoothStatus" class="status-badge" style="margin-left: 8px;">🔵 Bluetooth: Unknown</div>
        </div>
        
        <div id="personAlert" class="person-alert" style="display:none;">
            ⚠️ <span id="personAlertText">Person detected!</span> ⚠️
        </div>
        
        <div class="video-container">
            <img id="videoFeed" src="/video_feed" alt="Camera Feed">
        </div>
        
        <button class="emergency-btn" onclick="sendEmergency()">
            🚨 EMERGENCY BUTTON 🚨
        </button>
        
        <div class="card">
            <div class="stats-row">
                <div class="stat-box"><div class="stat-label">Persons</div><div class="stat-value" id="personCount">0</div></div>
                <div class="stat-box"><div class="stat-label">Vehicles</div><div class="stat-value" id="vehicleCount">0</div></div>
                <div class="stat-box"><div class="stat-label">FPS</div><div class="stat-value" id="fpsValue">0</div></div>
                <div class="stat-box"><div class="stat-label">Connected</div><div class="stat-value" id="mobileCount">0</div></div>
            </div>
            <div class="hazard-row">
                <div class="hazard-card" id="wallCard"><div>🧱 Wall</div><div id="wallStatus">Safe</div></div>
                <div class="hazard-card" id="stairsCard"><div>📐 Stairs</div><div id="stairsStatus">Safe</div></div>
                <div class="hazard-card" id="potholeCard"><div>🕳️ Pothole</div><div id="potholeStatus">Safe</div></div>
            </div>
            <button class="map-btn test-btn" onclick="testBuzzer()" style="margin-top: 12px;">🔊 Test Buzzer</button>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 8px;">👤 Detected Objects</h3>
            <div id="detectionList" class="detection-list">Waiting for detections...</div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 8px;">🔔 Alerts</h3>
            <div id="alertList" class="alert-list"><div class="alert-item"><div class="alert-time">System</div><div class="alert-title">✅ Ready</div><div class="alert-message">Person detection active with GPS tracking</div></div></div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 8px;">📍 Your GPS Location (Phone)</h3>
            <div id="map"></div>
            <div class="gps-status" id="gpsStatus">📍 Getting GPS location...</div>
            <div class="map-type-selector">
                <button class="map-type-btn" onclick="changeMapType('roadmap')">🗺️ Road</button>
                <button class="map-type-btn" onclick="changeMapType('satellite')">🛰️ Satellite</button>
                <button class="map-type-btn" onclick="changeMapType('hybrid')">🌍 Hybrid</button>
                <button class="map-type-btn" onclick="changeMapType('terrain')">⛰️ Terrain</button>
            </div>
            <div class="map-buttons">
                <div class="map-btn" onclick="openGoogleMaps()">🗺️ Open in Google Maps</div>
                <div class="map-btn" onclick="getDirections()">🧭 Get Directions</div>
                <div class="map-btn" onclick="refreshLocation()">🔄 Refresh</div>
                <div class="map-btn" onclick="shareLocation()">📤 Share</div>
            </div>
            <div id="locationInfo" class="location-text" onclick="openGoogleMaps()">Getting your GPS location...</div>
            <div id="coordinates" class="coordinates"></div>
            <div id="locationSource" class="source-badge"></div>
        </div>
    </div>
    
    <script src="https://maps.googleapis.com/maps/api/js?key={GOOGLE_MAPS_API_KEY}&callback=initGoogleMap" async defer></script>
    
    <script>
        let ws = null;
        let map = null;
        let marker = null;
        let deviceId = 'mobile_' + Math.random().toString(36).substr(2, 9);
        let currentLocation = null;
        let watchId = null;
        
        function initGoogleMap() {{
            const defaultLocation = {{ lat: 11.2745, lng: 77.5831 }};
            map = new google.maps.Map(document.getElementById('map'), {{
                center: defaultLocation,
                zoom: 16,
                mapTypeId: google.maps.MapTypeId.ROADMAP,
                mapTypeControl: true,
                streetViewControl: true,
                fullscreenControl: true,
                zoomControl: true
            }});
            
            marker = new google.maps.Marker({{
                position: defaultLocation,
                map: map,
                title: 'Your Location',
                icon: {{
                    url: 'https://maps.google.com/mapfiles/ms/icons/blue-dot.png',
                    scaledSize: new google.maps.Size(40, 40)
                }},
                animation: google.maps.Animation.BOUNCE
            }});
            
            const infoWindow = new google.maps.InfoWindow({{
                content: '<div style="padding: 8px;"><strong>📍 Your GPS Location</strong><br>Waiting for GPS...</div>'
            }});
            marker.addListener('click', () => infoWindow.open(map, marker));
            
            console.log('✅ Google Maps loaded');
            startGPS();
        }}
        
        function startGPS() {{
            if ("geolocation" in navigator) {{
                document.getElementById('gpsStatus').innerHTML = '📍 Getting GPS location...';
                
                // Watch position for continuous updates
                watchId = navigator.geolocation.watchPosition(
                    (position) => {{
                        const lat = position.coords.latitude;
                        const lng = position.coords.longitude;
                        const accuracy = position.coords.accuracy;
                        
                        currentLocation = {{
                            lat: lat,
                            lng: lng,
                            address: `GPS: ${{lat.toFixed(6)}}, ${{lng.toFixed(6)}}`,
                            accuracy: accuracy,
                            source: 'phone_gps'
                        }};
                        
                        updateGoogleMap(currentLocation);
                        sendGPSLocation(lat, lng, accuracy);
                        
                        const accuracyText = accuracy < 20 ? 'Excellent' : (accuracy < 50 ? 'Good' : 'Fair');
                        document.getElementById('gpsStatus').innerHTML = `📍 GPS Active - ${{accuracyText}} accuracy (${{Math.round(accuracy)}}m)`;
                        document.getElementById('locationInfo').innerHTML = `📍 GPS Location (accuracy: ${{Math.round(accuracy)}}m)`;
                        document.getElementById('coordinates').innerHTML = `📌 ${{lat.toFixed(6)}}, ${{lng.toFixed(6)}}`;
                        document.getElementById('locationSource').innerHTML = `📍 Source: PHONE GPS`;
                    }},
                    (error) => {{
                        console.error('GPS error:', error);
                        let errorMsg = 'GPS error: ';
                        switch(error.code) {{
                            case error.PERMISSION_DENIED:
                                errorMsg += 'Please allow location access';
                                break;
                            case error.POSITION_UNAVAILABLE:
                                errorMsg += 'Location unavailable';
                                break;
                            case error.TIMEOUT:
                                errorMsg += 'GPS timeout';
                                break;
                        }}
                        document.getElementById('gpsStatus').innerHTML = `⚠️ ${{errorMsg}}`;
                        document.getElementById('locationInfo').innerHTML = 'Using WiFi location...';
                    }},
                    {{
                        enableHighAccuracy: true,
                        timeout: 10000,
                        maximumAge: 0
                    }}
                );
            }} else {{
                document.getElementById('gpsStatus').innerHTML = '⚠️ GPS not supported on this device';
            }}
        }}
        
        function sendGPSLocation(lat, lng, accuracy) {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{
                    type: 'gps_location',
                    location: {{
                        lat: lat,
                        lng: lng,
                        accuracy: accuracy
                    }}
                }}));
            }}
        }}
        
        function changeMapType(type) {{
            if (!map) return;
            const mapTypes = {{
                'roadmap': google.maps.MapTypeId.ROADMAP,
                'satellite': google.maps.MapTypeId.SATELLITE,
                'hybrid': google.maps.MapTypeId.HYBRID,
                'terrain': google.maps.MapTypeId.TERRAIN
            }};
            map.setMapTypeId(mapTypes[type]);
            
            document.querySelectorAll('.map-type-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }}
        
        function updateGoogleMap(location) {{
            if (!map || !marker) return;
            
            const latLng = new google.maps.LatLng(location.lat, location.lng);
            map.setCenter(latLng);
            marker.setPosition(latLng);
            marker.setAnimation(google.maps.Animation.BOUNCE);
            setTimeout(() => marker.setAnimation(null), 2000);
            
            const infoWindow = new google.maps.InfoWindow({{
                content: `<div style="padding: 8px;">
                    <strong>📍 Your GPS Location</strong><br>
                    ${{location.lat.toFixed(6)}}, ${{location.lng.toFixed(6)}}<br>
                    Accuracy: ${{Math.round(location.accuracy || 10)}}m<br>
                    <button onclick="window.open('https://www.google.com/maps?q=${{location.lat}},${{location.lng}}', '_blank')" 
                            style="margin-top: 5px; padding: 5px 10px; background: #4caf50; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Open in Google Maps
                    </button>
                </div>`
            }});
            marker.addListener('click', () => infoWindow.open(map, marker));
        }}
        
        function openGoogleMaps() {{
            if (!currentLocation) {{
                alert('Getting GPS location...');
                return;
            }}
            const url = `https://www.google.com/maps?q=${{currentLocation.lat}},${{currentLocation.lng}}`;
            window.open(url, '_blank');
        }}
        
        function getDirections() {{
            if (!currentLocation) {{
                alert('Getting GPS location...');
                return;
            }}
            
            if (navigator.geolocation) {{
                navigator.geolocation.getCurrentPosition((position) => {{
                    const userLat = position.coords.latitude;
                    const userLng = position.coords.longitude;
                    const url = `https://www.google.com/maps/dir/${{userLat}},${{userLng}}/${{currentLocation.lat}},${{currentLocation.lng}}`;
                    window.open(url, '_blank');
                }}, () => {{
                    openGoogleMaps();
                }});
            }} else {{
                openGoogleMaps();
            }}
        }}
        
        function shareLocation() {{
            if (!currentLocation) return;
            
            const message = `📍 Smart Blind Stick Location\\n📌 Coordinates: ${{currentLocation.lat}}, ${{currentLocation.lng}}\\n🗺️ Google Maps: https://www.google.com/maps?q=${{currentLocation.lat}},${{currentLocation.lng}}`;
            
            if (navigator.share) {{
                navigator.share({{
                    title: 'Smart Blind Stick Location',
                    text: message,
                    url: `https://www.google.com/maps?q=${{currentLocation.lat}},${{currentLocation.lng}}`
                }});
            }} else {{
                navigator.clipboard.writeText(message);
                addAlert('Location Copied', 'GPS location copied to clipboard');
            }}
        }}
        
        function testBuzzer() {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type: 'test_buzzer' }}));
                addAlert('Buzzer Test', 'Testing Arduino buzzer...');
            }} else {{
                addAlert('Error', 'Not connected to laptop');
            }}
        }}
        
        function connectWebSocket() {{
            const wsUrl = `ws://${{window.location.hostname}}:8765`;
            console.log('Connecting to:', wsUrl);
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {{
                console.log('✅ WebSocket connected');
                document.getElementById('connectionStatus').innerHTML = '🟢 Connected - GPS Active';
                document.getElementById('connectionStatus').className = 'status-badge connected';
                ws.send(JSON.stringify({{ type: 'register', device_id: deviceId }}));
                addAlert('System', 'Connected with GPS tracking');
            }};
            
            ws.onmessage = (event) => {{
                try {{
                    const data = JSON.parse(event.data);
                    console.log('Received:', data);
                    updateUI(data);
                    if (data.type === 'emergency') {{
                        handleEmergency(data);
                    }}
                }} catch(e) {{
                    console.error('Parse error:', e);
                }}
            }};
            
            ws.onclose = () => {{
                console.log('WebSocket disconnected');
                document.getElementById('connectionStatus').innerHTML = '🔴 Disconnected';
                document.getElementById('connectionStatus').className = 'status-badge disconnected';
                setTimeout(connectWebSocket, 3000);
            }};
        }}
        
        function updateUI(data) {{
            if (data.person_count !== undefined) {{
                document.getElementById('personCount').innerText = data.person_count;
                
                // Show person alert if persons detected
                if (data.person_count > 0) {{
                    document.getElementById('personAlert').style.display = 'block';
                    if (data.detections && data.detections.length > 0) {{
                        const person = data.detections.find(d => d.class === 'person');
                        if (person) {{
                            let alertText = `⚠️ Person detected ${{person.direction}}!`;
                            if (person.distance === 'very close') alertText = `⚠️ WARNING! Person ${{person.direction}}, VERY CLOSE!`;
                            else if (person.distance === 'close') alertText = `⚠️ Person ${{person.direction}}, close!`;
                            document.getElementById('personAlertText').innerText = alertText;
                        }}
                    }}
                }} else {{
                    document.getElementById('personAlert').style.display = 'none';
                }}
            }}
            if (data.vehicle_count !== undefined) {{
                document.getElementById('vehicleCount').innerText = data.vehicle_count;
            }}
            if (data.fps !== undefined) {{
                document.getElementById('fpsValue').innerText = data.fps;
            }}
            if (data.connected_clients !== undefined) {{
                document.getElementById('mobileCount').innerText = data.connected_clients;
            }}
            if (data.bluetooth_connected !== undefined) {{
                const btStatus = data.bluetooth_connected ? '🟢 Connected' : '🔴 Disconnected';
                document.getElementById('bluetoothStatus').innerHTML = `🔵 Bluetooth: ${{btStatus}}`;
                document.getElementById('bluetoothStatus').className = data.bluetooth_connected ? 'status-badge connected' : 'status-badge disconnected';
            }}
            
            if (data.walls) {{
                updateHazard('wall', data.walls.detected, data.walls.distance);
            }}
            if (data.stairs) {{
                updateHazard('stairs', data.stairs.detected, data.stairs.confidence);
            }}
            if (data.potholes) {{
                updateHazard('pothole', data.potholes.detected, data.potholes.confidence);
            }}
            
            if (data.detections) {{
                updateDetections(data.detections);
            }}
        }}
        
        function updateHazard(type, detected, info) {{
            const card = document.getElementById(`${{type}}Card`);
            const status = document.getElementById(`${{type}}Status`);
            if (detected) {{
                card.classList.add('active');
                let displayInfo = info;
                if (typeof info === 'number') displayInfo = info + '%';
                status.innerHTML = `⚠️ ${{displayInfo}}`;
                status.style.color = '#ff4444';
            }} else {{
                card.classList.remove('active');
                status.innerHTML = 'Safe';
                status.style.color = '';
            }}
        }}
        
        function updateDetections(detections) {{
            const container = document.getElementById('detectionList');
            if (!detections || detections.length === 0) {{
                container.innerHTML = '<div style="text-align:center;padding:20px;opacity:0.5;">🔍 No objects detected</div>';
                return;
            }}
            
            let html = '';
            detections.slice(0, 10).forEach(obj => {{
                const isDanger = obj.distance === 'very close';
                const isWarning = obj.distance === 'close';
                const emojiMap = {{
                    'person': '👤', 'vehicle': '🚗', 'car': '🚗', 'bicycle': '🚲',
                    'motorcycle': '🏍️', 'bus': '🚌', 'truck': '🚚'
                }};
                const emoji = emojiMap[obj.class] || '📦';
                const dangerClass = isDanger ? 'danger' : (isWarning ? 'warning' : '');
                const color = isDanger ? '#ff4444' : (isWarning ? '#ff9800' : '#4caf50');
                
                html += `<div class="detection-item ${{dangerClass}}">
                    <div>
                        <strong>${{emoji}} ${{obj.class}}</strong><br>
                        <small style="opacity:0.7;">${{obj.direction}} • ${{obj.distance}}</small>
                    </div>
                    <div style="color:${{color}};">${{Math.round(obj.confidence * 100)}}%</div>
                </div>`;
            }});
            container.innerHTML = html;
        }}
        
        function refreshLocation() {{
            if (navigator.geolocation) {{
                navigator.geolocation.getCurrentPosition((position) => {{
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    sendGPSLocation(lat, lng, position.coords.accuracy);
                    addAlert('Location', 'GPS location updated');
                }});
            }}
        }}
        
        function handleEmergency(data) {{
            addAlert(data.title, data.message, true);
            
            try {{
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.frequency.value = 880;
                gain.gain.value = 0.5;
                osc.start();
                gain.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + 2);
                setTimeout(() => osc.stop(), 2000);
            }} catch(e) {{}}
            
            if (navigator.vibrate) {{
                navigator.vibrate([500, 300, 500, 300, 500]);
            }}
            
            if (Notification.permission === 'granted') {{
                new Notification(data.title, {{ 
                    body: data.message,
                    icon: '🔴',
                    requireInteraction: true
                }});
            }}
            
            if (data.maps_url) {{
                setTimeout(() => {{
                    if (confirm('🚨 EMERGENCY ALERT!\\n\\nOpen Google Maps for navigation?')) {{
                        window.open(data.maps_url, '_blank');
                    }}
                }}, 1000);
            }}
        }}
        
        async function sendEmergency() {{
            if (confirm('⚠️ EMERGENCY ALERT ⚠️\\n\\nSend immediate alert with your GPS location?')) {{
                try {{
                    const response = await fetch('/emergency', {{ method: 'POST' }});
                    const data = await response.json();
                    if (data.status === 'success') {{
                        addAlert('🚨 EMERGENCY SENT', 'Alert sent with GPS location!', true);
                        if (navigator.vibrate) navigator.vibrate([500, 500, 500]);
                    }} else {{
                        addAlert('Error', 'Failed to send emergency alert');
                    }}
                }} catch(e) {{
                    console.error('Emergency error:', e);
                    addAlert('Error', 'Failed to send emergency alert');
                }}
            }}
        }}
        
        function addAlert(title, message, isEmergency = false) {{
            const container = document.getElementById('alertList');
            const time = new Date().toLocaleTimeString();
            const div = document.createElement('div');
            div.className = `alert-item ${{isEmergency ? 'emergency' : ''}}`;
            div.innerHTML = `<div class="alert-time">${{time}}</div><div class="alert-title">${{title}}</div><div class="alert-message">${{message}}</div>`;
            container.insertBefore(div, container.firstChild);
            while (container.children.length > 15) container.removeChild(container.lastChild);
        }}
        
        async function updateStats() {{
            try {{
                const res = await fetch('/stats');
                const data = await res.json();
                if (data.connected_clients !== undefined) {{
                    document.getElementById('mobileCount').innerText = data.connected_clients;
                }}
            }} catch(e) {{}}
        }}
        
        if (Notification.permission === 'default') {{
            Notification.requestPermission();
        }}
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'e' || e.key === 'E') sendEmergency();
        }});
        
        setInterval(updateStats, 5000);
        connectWebSocket();
        
        console.log('✅ Mobile assistant ready - GPS tracking active');
    </script>
</body>
</html>
'''

blind_stick = None

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    if blind_stick:
        return Response(blind_stick.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera not initialized", 500

@app.route('/stats')
def stats():
    if blind_stick:
        return jsonify({
            'connected_clients': len(blind_stick.clients),
            'person_count': blind_stick.person_count,
            'vehicle_count': blind_stick.vehicle_count,
            'fps': blind_stick.fps,
            'emergency': blind_stick.emergency_mode,
            'location': blind_stick.current_location,
            'detection_count': blind_stick.detection_count,
            'bluetooth_connected': bluetooth_serial is not None and bluetooth_serial.is_open,
            'last_person_location': blind_stick.last_person_location
        })
    return jsonify({'connected_clients': 0, 'person_count': 0, 'fps': 0})

@app.route('/emergency', methods=['POST'])
def emergency():
    if blind_stick:
        blind_stick.emergency_mode = True
        blind_stick.speak("EMERGENCY! Help needed immediately!", "emergency")
        
        async def send():
            await blind_stick.send_emergency(blind_stick.current_location, blind_stick.person_count)
        
        if hasattr(blind_stick, 'ws_loop'):
            asyncio.run_coroutine_threadsafe(send(), blind_stick.ws_loop)
        
        def reset():
            time.sleep(30)
            blind_stick.emergency_mode = False
        threading.Thread(target=reset, daemon=True).start()
        
        maps_url = f"https://www.google.com/maps?q={blind_stick.current_location['lat']},{blind_stick.current_location['lng']}"
        return jsonify({'status': 'success', 'maps_url': maps_url})
    return jsonify({'status': 'error'}), 500

@app.route('/database/stats')
def database_stats():
    if blind_stick:
        return jsonify(blind_stick.db.get_stats())
    return jsonify({}), 500

@app.route('/database/detections')
def database_detections():
    if blind_stick:
        detections = blind_stick.db.get_recent_detections(50)
        for d in detections:
            if '_id' in d:
                d['_id'] = str(d['_id'])
        return jsonify(detections)
    return jsonify([]), 500

@app.route('/database/alerts')
def database_alerts():
    if blind_stick:
        alerts = blind_stick.db.get_recent_alerts(50)
        for a in alerts:
            if '_id' in a:
                a['_id'] = str(a['_id'])
        return jsonify(alerts)
    return jsonify([]), 500

@app.route('/database/analytics')
def database_analytics():
    if blind_stick:
        analytics = {
            'detections_by_type': blind_stick.db.get_detections_by_type(),
            'alerts_by_type': blind_stick.db.get_alerts_by_type(),
            'stats': blind_stick.db.get_stats()
        }
        return jsonify(analytics)
    return jsonify({}), 500

if __name__ == "__main__":
    # Initialize Bluetooth first
    init_bluetooth()
    
    mongodb_uri = "mongodb://localhost:27017/"
    
    blind_stick = SmartBlindStick(mongodb_uri)
    blind_stick_thread = threading.Thread(target=blind_stick.run, daemon=True)
    blind_stick_thread.start()
    time.sleep(2)
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print("🌐 SMART BLIND STICK SYSTEM WITH ENHANCED FEATURES!")
    print("="*60)
    print(f"📱 MOBILE PHONE: http://{local_ip}:5000")
    print(f"🗄️ DATABASE: MongoDB (blind_stick_db)")
    print(f"🗺️ Google Maps API Key: {GOOGLE_MAPS_API_KEY[:10]}...")
    print(f"🔵 BLUETOOTH: {'Connected ✓' if bluetooth_serial else 'Not Connected ✗'}")
    print("\n🎯 NEW ENHANCEMENTS:")
    print("   ✅ Person detection with VOICE alerts")
    print("   ✅ Direction tracking (left/right/straight)")
    print("   ✅ Distance estimation (very close/close/far)")
    print("   ✅ Phone GPS location tracking")
    print("   ✅ Person location saved to database")
    print("\n🎯 GOOGLE MAPS FEATURES:")
    print("   • Phone GPS shows YOUR exact location")
    print("   • Multiple map views (Road, Satellite, Hybrid, Terrain)")
    print("   • Street View available")
    print("   • Directions from your location")
    print("   • Share your location")
    print("\n🎯 PERSON DETECTION VOICE EXAMPLES:")
    print("   • 'Warning! Person on your left, very close!'")
    print("   • 'Person detected straight ahead'")
    print("   • 'There is a person on your right at [location]'")
    print("\n💡 Press 'E' key for emergency alert")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)