"""
CENTRAL SERVER - SISTEM ABSENSI FACE RECOGNITION
Backend API untuk 3 Client terpisah:
1. Web Absensi (Alat di Sekolah)
2. Web Guru (Dashboard Admin)
3. Web Siswa (Self Registration)
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from scipy.spatial import distance
import pickle
import json
import os
import base64
from datetime import datetime, timedelta
import pandas as pd
import hashlib
import secrets

# ==================== KONFIGURASI ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'central_server_secret_key_2024'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Paths
DATASET_PATH = "dataset/faces/"
MODEL_PATH = "models/face_recognition_model.h5"
ENCODINGS_PATH = "models/face_encodings.pkl"
STUDENTS_DB = "database/students.json"
ATTENDANCE_DB = "database/attendance.csv"
PENDING_REGISTRATIONS = "database/pending_registrations.json"

# Parameters
IMG_SIZE = (224, 224)
EAR_THRESHOLD = 0.25
BLINK_CONSEC_FRAMES = 2

# Create directories
for directory in ["dataset/faces", "models", "database", "client1", "client2", "client3"]:
    os.makedirs(directory, exist_ok=True)

# ==================== DATABASE HELPERS ====================
class Database:
    @staticmethod
    def load_students():
        """Load data siswa dari JSON"""
        if os.path.exists(STUDENTS_DB):
            with open(STUDENTS_DB, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    @staticmethod
    def save_students(data):
        """Simpan data siswa ke JSON"""
        with open(STUDENTS_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    @staticmethod
    def load_pending():
        """Load pendaftaran yang pending approval"""
        if os.path.exists(PENDING_REGISTRATIONS):
            with open(PENDING_REGISTRATIONS, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    @staticmethod
    def save_pending(data):
        """Simpan pendaftaran pending"""
        with open(PENDING_REGISTRATIONS, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    @staticmethod
    def add_attendance(record):
        """Tambah record kehadiran"""
        df = pd.DataFrame([record])
        if os.path.exists(ATTENDANCE_DB):
            df.to_csv(ATTENDANCE_DB, mode='a', header=False, index=False)
        else:
            df.to_csv(ATTENDANCE_DB, mode='w', header=True, index=False)
    
    @staticmethod
    def get_attendance(date=None):
        """Get attendance records"""
        if not os.path.exists(ATTENDANCE_DB):
            return []
        
        df = pd.read_csv(ATTENDANCE_DB)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        if date:
            df = df[df['Timestamp'].dt.date == pd.to_datetime(date).date()]
        
        return df.to_dict('records')
    
    @staticmethod
    def check_already_present_today(nis):
        """Cek apakah sudah absen hari ini"""
        if not os.path.exists(ATTENDANCE_DB):
            return False
        
        df = pd.read_csv(ATTENDANCE_DB)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        today = pd.Timestamp.now().date()
        
        today_records = df[(df['Timestamp'].dt.date == today) & (df['NIS'] == nis)]
        return len(today_records) > 0

# ==================== LIVENESS DETECTION ====================
class LivenessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.blink_counter = 0
        self.total_blinks = 0
        
    def calculate_ear(self, eye_landmarks):
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)
    
    def detect_blink(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        is_live = False
        ear = 0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            left_eye = []
            right_eye = []
            
            for idx in self.LEFT_EYE:
                landmark = face_landmarks.landmark[idx]
                left_eye.append((landmark.x * w, landmark.y * h))
                
            for idx in self.RIGHT_EYE:
                landmark = face_landmarks.landmark[idx]
                right_eye.append((landmark.x * w, landmark.y * h))
            
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            if ear < EAR_THRESHOLD:
                self.blink_counter += 1
            else:
                if self.blink_counter >= BLINK_CONSEC_FRAMES:
                    self.total_blinks += 1
                self.blink_counter = 0
            
            if self.total_blinks >= 2:
                is_live = True
                
        return is_live, ear, self.total_blinks
    
    def reset(self):
        self.blink_counter = 0
        self.total_blinks = 0

# ==================== FACE RECOGNITION ====================
class FaceRecognitionCNN:
    def __init__(self):
        self.model = None
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.load_trained_model()
        
    def load_trained_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = load_model(MODEL_PATH)
            print("✅ Model CNN loaded successfully")
            return True
        print("⚠️ Model not found. Train model first.")
        return False
    
    def extract_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        return faces
    
    def preprocess_face(self, face_img):
        face_resized = cv2.resize(face_img, IMG_SIZE)
        face_normalized = face_resized / 255.0
        return np.expand_dims(face_normalized, axis=0)
    
    def recognize_face(self, face_img, threshold=0.7):
        if self.model is None:
            return None, 0.0
        
        preprocessed = self.preprocess_face(face_img)
        predictions = self.model.predict(preprocessed, verbose=0)
        
        max_prob = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        if max_prob >= threshold:
            return int(predicted_class), float(max_prob)
        return None, float(max_prob)

# ==================== GLOBAL INSTANCES ====================
liveness_detectors = {}  # Per client session
face_recognition = FaceRecognitionCNN()
registration_sessions = {}  # Temporary storage untuk registrasi

# ==================== AUTH & USER MANAGEMENT ====================
STUDENTS_MASTER_DB = "database/students_master.json"  # Database master siswa sekolah
TEACHERS_DB = "database/teachers.json"

class UserAuth:
    @staticmethod
    def load_master_students():
        """Load database master siswa sekolah (dari sistem akademik)"""
        if os.path.exists(STUDENTS_MASTER_DB):
            with open(STUDENTS_MASTER_DB, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Default data siswa untuk testing (biasanya import dari sistem akademik)
        default_students = {
            "15230009": {
                "nis": "15230009",
                "name": "Efriza Taufiqurrohman",
                "class": "X MIPA 1",
                "email": "efriza@example.com",
                "phone": "081234567890",
                "face_registered": False,
                "created_at": "2024-01-10"
            },
            "15230349": {
                "nis": "15230349",
                "name": "Denis Pratama Putra",
                "class": "X MIPA 1",
                "email": "denis@example.com",
                "phone": "081234567891",
                "face_registered": False,
                "created_at": "2024-01-10"
            },
            "15230096": {
                "nis": "15230096",
                "name": "Nurmalisa Al Hawa",
                "class": "X MIPA 2",
                "email": "nurmalisa@example.com",
                "phone": "081234567892",
                "face_registered": False,
                "created_at": "2024-01-10"
            }
        }
        
        UserAuth.save_master_students(default_students)
        return default_students
    
    @staticmethod
    def save_master_students(data):
        """Simpan database master siswa"""
        with open(STUDENTS_MASTER_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    @staticmethod
    def load_teachers():
        """Load data walikelas"""
        if os.path.exists(TEACHERS_DB):
            with open(TEACHERS_DB, 'r', encoding='utf-8') as f:
                return json.load(f)
        # Default teachers untuk testing
        return {
            "guru001": {
                "username": "guru001",
                "password": "password123",
                "name": "Budi Santoso",
                "class": "X MIPA 1"
            },
            "guru002": {
                "username": "guru002",
                "password": "password123",
                "name": "Siti Rahayu",
                "class": "X MIPA 2"
            },
            "guru003": {
                "username": "guru003",
                "password": "password123",
                "name": "Ahmad Hidayat",
                "class": "XI IPA 1"
            }
        }
    
    @staticmethod
    def save_teachers(data):
        """Simpan data walikelas"""
        with open(TEACHERS_DB, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    @staticmethod
    def update_face_status(nis, status=True):
        """Update status registrasi wajah siswa"""
        students = UserAuth.load_master_students()
        if nis in students:
            students[nis]["face_registered"] = status
            UserAuth.save_master_students(students)
            return True
        return False
    
    @staticmethod
    def authenticate_student(nis, name):
        """Autentikasi siswa dengan NIS dan Nama"""
        students = UserAuth.load_master_students()
        
        if nis in students:
            student = students[nis]
            # Case-insensitive comparison untuk nama
            if student['name'].lower() == name.lower():
                return {
                    'success': True,
                    'student': student
                }
        
        return {
            'success': False,
            'message': 'NIS atau Nama tidak cocok dengan data sekolah'
        }

# Initialize databases
if not os.path.exists(TEACHERS_DB):
    UserAuth.save_teachers(UserAuth.load_teachers())

if not os.path.exists(STUDENTS_MASTER_DB):
    UserAuth.load_master_students()  # Will create default data

# ==================== REST API ENDPOINTS ====================

# Home - Info API
@app.route('/')
def index():
    return jsonify({
        'status': 'running',
        'service': 'Central Face Recognition Server',
        'version': '2.0.0',
        'endpoints': {
            'students': '/api/students',
            'attendance': '/api/attendance',
            'stats': '/api/stats',
            'pending': '/api/pending-registrations',
            'auth': '/api/auth/*'
        }
    })

# ==================== AUTH ENDPOINTS ====================

@app.route('/api/auth/authenticate-student', methods=['POST'])
def authenticate_student_api():
    """Autentikasi siswa HANYA dengan NIS dan Nama (untuk Client 3)"""
    data = request.json
    nis = data.get('nis')
    name = data.get('name')

    if not nis or not name:
        return jsonify({
            'success': False,
            'message': 'NIS dan Nama harus diisi'
        }), 400

    # Memanggil fungsi yang SUDAH ADA di server.py
    auth_result = UserAuth.authenticate_student(nis, name)
    
    if auth_result['success']:
        # Mengirim kembali data 'student' lengkap
        return jsonify({
            'success': True,
            'student': auth_result['student'], 
            'message': 'Login berhasil'
        })
    else:
        return jsonify({
            'success': False,
            'message': auth_result.get('message', 'NIS atau Nama salah')
        }), 401

@app.route('/api/auth/login-student', methods=['POST'])
def login_student():
    """Login untuk siswa"""
    data = request.json
    nis = data.get('nis')
    password = data.get('password')
    
    users = UserAuth.load_users()
    
    if nis in users and users[nis]['password'] == password:
        return jsonify({
            'success': True,
            'user': {
                'nis': users[nis]['nis'],
                'name': users[nis]['name'],
                'face_registered': users[nis].get('face_registered', False)
            },
            'message': 'Login berhasil'
        })
    
    return jsonify({
        'success': False,
        'message': 'NIS atau password salah'
    }), 401

@app.route('/api/auth/register-student', methods=['POST'])
def register_student():
    """Register akun siswa baru"""
    data = request.json
    nis = data.get('nis')
    name = data.get('name')
    password = data.get('password')
    
    if not nis or not name or not password:
        return jsonify({
            'success': False,
            'message': 'Data tidak lengkap'
        }), 400
    
    users = UserAuth.load_users()
    
    if nis in users:
        return jsonify({
            'success': False,
            'message': 'NIS sudah terdaftar'
        }), 400
    
    UserAuth.create_student_account(nis, name, password)
    
    return jsonify({
        'success': True,
        'message': 'Registrasi berhasil! Silakan login.'
    })

@app.route('/api/auth/login-teacher', methods=['POST'])
def login_teacher():
    """Login untuk walikelas"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    teachers = UserAuth.load_teachers()
    
    if username in teachers and teachers[username]['password'] == password:
        return jsonify({
            'success': True,
            'teacher': {
                'username': teachers[username]['username'],
                'name': teachers[username]['name'],
                'class': teachers[username]['class']
            },
            'message': 'Login berhasil'
        })
    
    return jsonify({
        'success': False,
        'message': 'Username atau password salah'
    }), 401

@app.route('/api/auth/check-face-status/<nis>', methods=['GET'])
def check_face_status(nis):
    """Cek status registrasi wajah siswa"""
    users = UserAuth.load_users()
    
    if nis in users:
        return jsonify({
            'success': True,
            'face_registered': users[nis].get('face_registered', False)
        })
    
    return jsonify({
        'success': False,
        'message': 'User tidak ditemukan'
    }), 404

# Get all students
@app.route('/api/students', methods=['GET'])
def get_students():
    students = Database.load_students()
    return jsonify({
        'success': True,
        'count': len(students),
        'students': list(students.values())
    })

# Get single student
@app.route('/api/students/<nis>', methods=['GET'])
def get_student(nis):
    students = Database.load_students()
    if nis in students:
        return jsonify({
            'success': True,
            'student': students[nis]
        })
    return jsonify({
        'success': False,
        'message': 'Student not found'
    }), 404

# Get attendance records
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    date = request.args.get('date')  # Format: YYYY-MM-DD
    records = Database.get_attendance(date)
    return jsonify({
        'success': True,
        'count': len(records),
        'records': records
    })

# Get today's attendance (dengan filter kelas)
@app.route('/api/attendance/today', methods=['GET'])
def get_today_attendance():
    class_filter = request.args.get('class')  # Filter by class
    
    today = datetime.now().strftime('%Y-%m-%d')
    records = Database.get_attendance(today)
    
    # Filter by class if provided
    if class_filter:
        records = [r for r in records if r.get('Kelas') == class_filter]
    
    return jsonify({
        'success': True,
        'date': today,
        'count': len(records),
        'records': records,
        'class': class_filter
    })

# Get statistics (dengan filter kelas untuk walikelas)
@app.route('/api/stats', methods=['GET'])
def get_stats():
    class_filter = request.args.get('class')  # Filter by class
    
    students = Database.load_students()
    
    # Filter students by class if provided
    if class_filter:
        filtered_students = {k: v for k, v in students.items() if v.get('class') == class_filter}
    else:
        filtered_students = students
    
    total_students = len(filtered_students)
    
    today = datetime.now().strftime('%Y-%m-%d')
    today_records = Database.get_attendance(today)
    
    # Filter attendance by class if provided
    if class_filter:
        today_records = [r for r in today_records if r.get('Kelas') == class_filter]
    
    today_attendance = len(today_records)
    
    all_records = Database.get_attendance()
    if class_filter:
        all_records = [r for r in all_records if r.get('Kelas') == class_filter]
    
    total_attendance = len(all_records)
    
    attendance_rate = 0
    if total_students > 0:
        attendance_rate = round((today_attendance / total_students) * 100, 2)
    
    return jsonify({
        'success': True,
        'totalStudents': total_students,
        'todayAttendance': today_attendance,
        'totalAttendance': total_attendance,
        'attendanceRate': attendance_rate,
        'class': class_filter
    })

# Get pending registrations (untuk approval guru)
@app.route('/api/pending-registrations', methods=['GET'])
def get_pending_registrations():
    pending = Database.load_pending()
    return jsonify({
        'success': True,
        'count': len(pending),
        'registrations': pending
    })

# Approve registration (guru) - Update user face status
import shutil # Pastikan import shutil ada di ATAS file, bukan di dalam fungsi

# ... (kode-kode lain) ...

@app.route('/api/approve-registration/<request_id>', methods=['POST'])
def approve_registration(request_id):
    pending = Database.load_pending()
    
    for i, reg in enumerate(pending):
        if reg['request_id'] == request_id:
            try:
                # --- PERBAIKAN LOGIKA ID ---
                students = Database.load_students()
                
                # Cek apakah NIS sudah ada untuk menentukan ID
                if reg['nis'] in students and 'student_id' in students[reg['nis']]:
                    student_id = students[reg['nis']]['student_id'] # Gunakan ID yang ada
                else:
                    student_id = len(students) # Buat ID baru
                # --- SELESAI PERBAIKAN ID ---

                students[reg['nis']] = {
                    'student_id': student_id, # Gunakan ID yang sudah benar
                    'nis': reg['nis'],
                    'name': reg['name'],
                    'class': reg['class'],
                    'email': reg.get('email', ''),
                    'phone': reg.get('phone', ''),
                    'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'approved_by': request.json.get('approved_by', 'admin'),
                    'status': 'active'
                }
                Database.save_students(students)
                
                # Update status di master database
                UserAuth.update_face_status(reg['nis'], True)
                
                # Pindahkan foto
                pending_dir = f"dataset/pending/{request_id}/"
                # Gunakan 'student_id' yang sudah benar
                approved_dir = f"dataset/faces/{student_id}_{reg['name'].replace(' ', '_')}/" 
                
                if os.path.exists(pending_dir):
                    os.makedirs(approved_dir, exist_ok=True)
                    
                    # Hapus file lama di folder approved (jika ada, untuk re-registrasi)
                    for file in os.listdir(approved_dir):
                        if file.endswith(('.jpg', '.png')):
                            os.remove(os.path.join(approved_dir, file))
                    
                    # Pindahkan file baru
                    for file in os.listdir(pending_dir):
                        if file.endswith(('.jpg', '.png')):
                            shutil.move(
                                os.path.join(pending_dir, file),
                                os.path.join(approved_dir, file)
                            )
                    # Hapus folder pending setelah selesai
                    shutil.rmtree(pending_dir) 
                
                # Hapus dari pending
                pending.pop(i)
                Database.save_pending(pending)
                
                return jsonify({
                    'success': True,
                    'message': f'Registration approved for {reg["name"]}'
                })
            
            except Exception as e:
                print(f"Error approving registration: {e}")
                return jsonify({
                    'success': False,
                    'message': f'Server error: {e}'
                }), 500
    
    return jsonify({
        'success': False,
        'message': 'Registration not found'
    }), 404

# Reject registration
@app.route('/api/reject-registration/<request_id>', methods=['POST'])
def reject_registration(request_id):
    pending = Database.load_pending()
    
    for i, reg in enumerate(pending):
        if reg['request_id'] == request_id:
            pending.pop(i)
            Database.save_pending(pending)
            
            return jsonify({
                'success': True,
                'message': 'Registration rejected'
            })
    
    return jsonify({
        'success': False,
        'message': 'Registration not found'
    }), 404

# ==================== SOCKETIO EVENTS ====================

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    print(f'✅ Client connected: {client_id}')
    emit('connected', {'client_id': client_id, 'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    if client_id in liveness_detectors:
        del liveness_detectors[client_id]
    if client_id in registration_sessions:
        del registration_sessions[client_id]
    print(f'❌ Client disconnected: {client_id}')

@socketio.on('join_room')
def handle_join_room(data):
    room = data.get('room')  # 'absensi', 'guru', or 'siswa'
    join_room(room)
    emit('room_joined', {'room': room}, room=request.sid)

# ==================== ABSENSI (CLIENT 1) ====================

@socketio.on('start_attendance')
def handle_start_attendance():
    client_id = request.sid
    liveness_detectors[client_id] = LivenessDetector()
    
    emit('attendance_status', {
        'status': 'started',
        'message': 'Posisikan wajah dan kedipkan mata 2x'
    })

@socketio.on('process_attendance_frame')
def handle_attendance_frame(data):
    client_id = request.sid
    
    try:
        # Decode frame
        img_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Liveness detection
        if client_id not in liveness_detectors:
            liveness_detectors[client_id] = LivenessDetector()
        
        detector = liveness_detectors[client_id]
        is_live, ear, blink_count = detector.detect_blink(frame)
        
        emit('blink_update', {
            'blinkCount': blink_count,
            'ear': float(ear),
            'isLive': is_live
        })
        
        if is_live:
            # Face recognition
            faces = face_recognition.extract_face(frame)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                
                student_id, confidence = face_recognition.recognize_face(face_img)
                
                if student_id is not None:
                    students = Database.load_students()
                    
                    # Cari student by ID
                    student_info = None
                    for nis, data in students.items():
                        if data.get('student_id') == student_id:
                            student_info = data
                            student_info['nis'] = nis
                            break
                    
                    if student_info:
                        # Check duplicate
                        if Database.check_already_present_today(student_info['nis']):
                            emit('attendance_result', {
                                'status': 'duplicate',
                                'message': f"{student_info['name']} sudah absen hari ini!",
                                'student': student_info
                            })
                        else:
                            # Save attendance
                            record = {
                                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'NIS': student_info['nis'],
                                'Nama': student_info['name'],
                                'Kelas': student_info['class'],
                                'Status': 'Hadir',
                                'Metode': 'Face Recognition + Liveness',
                                'Confidence': f"{confidence:.2f}"
                            }
                            Database.add_attendance(record)
                            
                            emit('attendance_result', {
                                'status': 'success',
                                'message': f"Selamat datang, {student_info['name']}!",
                                'student': student_info,
                                'confidence': confidence
                            })
                            
                            # Broadcast to teacher dashboard
                            socketio.emit('new_attendance', record, room='guru')
                    else:
                        emit('attendance_result', {
                            'status': 'error',
                            'message': 'Data siswa tidak ditemukan!'
                        })
                else:
                    emit('attendance_result', {
                        'status': 'unknown',
                        'message': 'Wajah tidak dikenali. Pastikan Anda sudah terdaftar.',
                        'confidence': confidence
                    })
        
    except Exception as e:
        print(f"Error processing attendance: {e}")
        emit('error', {'message': str(e)})

# ==================== REGISTRASI SISWA (CLIENT 3) ====================

@socketio.on('start_student_registration')
def handle_student_registration(data):
    client_id = request.sid
    
    # Validasi data
    required_fields = ['name', 'nis', 'class']
    if not all(field in data for field in required_fields):
        emit('registration_error', {'message': 'Data tidak lengkap!'})
        return
    
    # Check duplicate NIS
    students = Database.load_students()
    if data['nis'] in students:
        emit('registration_error', {'message': 'NIS sudah terdaftar!'})
        return
    
    # Check if already in pending
    pending = Database.load_pending()
    for reg in pending:
        if reg['nis'] == data['nis']:
            emit('registration_error', {'message': 'Pendaftaran Anda masih dalam proses approval!'})
            return
    
    # Cek apakah user sudah punya akun dan sudah registrasi wajah
    users = UserAuth.load_master_students()
    if data['nis'] in users and users[data['nis']].get('face_registered', False):
        emit('registration_error', {'message': 'Wajah Anda sudah terdaftar!'})
        return
    
    # Initialize session
    liveness_detectors[client_id] = LivenessDetector()
    registration_sessions[client_id] = {
        'data': data,
        'captured_faces': [],
        'request_id': secrets.token_hex(8)
    }
    
    emit('registration_status', {
        'status': 'started',
        'message': 'Kedipkan mata 2x untuk verifikasi'
    })

@socketio.on('process_registration_frame')
def handle_registration_frame(data):
    client_id = request.sid
    
    try:
        if client_id not in registration_sessions:
            return
        
        # Decode frame
        img_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Liveness detection
        detector = liveness_detectors[client_id]
        is_live, ear, blink_count = detector.detect_blink(frame)
        
        emit('blink_update', {
            'blinkCount': blink_count,
            'ear': float(ear),
            'isLive': is_live
        })
        
        if is_live:
            session = registration_sessions[client_id]
            captured = session['captured_faces']
            
            # Capture faces
            faces = face_recognition.extract_face(frame)
            if len(faces) > 0 and len(captured) < 20:
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                captured.append(face_img)
                
                emit('capture_update', {
                    'capturedCount': len(captured),
                    'total': 20
                })
                
                if len(captured) >= 20:
                    # Complete registration
                    complete_student_registration(client_id)
        
    except Exception as e:
        print(f"Error processing registration: {e}")
        emit('error', {'message': str(e)})

def complete_student_registration(client_id):
    session = registration_sessions[client_id]
    student_data = session['data']
    captured_faces = session['captured_faces']
    request_id = session['request_id']
    
    # Save to pending
    pending = Database.load_pending()
    pending.append({
        'request_id': request_id,
        'nis': student_data['nis'],
        'name': student_data['name'],
        'class': student_data['class'],
        'email': student_data.get('email', ''),
        'phone': student_data.get('phone', ''),
        'captured_faces': len(captured_faces),
        'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'pending'
    })
    Database.save_pending(pending)
    
    # Save face images temporarily
    temp_dir = f"dataset/pending/{request_id}/"
    os.makedirs(temp_dir, exist_ok=True)
    for i, face in enumerate(captured_faces):
        cv2.imwrite(f"{temp_dir}/face_{i}.jpg", face)
    
    # Notify student
    socketio.emit('registration_complete', {
        'status': 'pending',
        'message': 'Pendaftaran berhasil! Menunggu approval dari guru.',
        'request_id': request_id
    }, room=client_id)
    
    # Notify teacher dashboard
    socketio.emit('new_registration_request', {
        'request_id': request_id,
        'student': student_data,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, room='guru')
    
    # Cleanup
    del registration_sessions[client_id]
    del liveness_detectors[client_id]

# ==================== RUN SERVER ====================
if __name__ == '__main__':
    print("="*70)
    print("🚀 CENTRAL SERVER - SISTEM ABSENSI FACE RECOGNITION")
    print("="*70)
    print("📡 Server running at: http://localhost:5000")
    print("📱 Client 1 (Absensi): http://localhost:5001")
    print("👨‍🏫 Client 2 (Guru): http://localhost:5002")
    print("🎓 Client 3 (Siswa): http://localhost:5003")
    print("="*70)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)