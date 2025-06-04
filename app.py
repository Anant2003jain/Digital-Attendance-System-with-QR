import os
import warnings

# Disable TensorFlow warnings and set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qrcode as qc
import csv
import base64
import io
import uuid
import hashlib
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
from functools import wraps

# Import TensorFlow after setting environment variables
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)

# Make datetime available to all templates
@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

# Create necessary directories if they don't exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/qrcodes', exist_ok=True)
os.makedirs('static/charts', exist_ok=True)
os.makedirs('static/models', exist_ok=True)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'admin' or 'user'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    enrollment_no = db.Column(db.String(20), unique=True, nullable=False)
    name_id = db.Column(db.String(100), unique=True, nullable=False)
    qr_code_path = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    attendances = db.relationship('Attendance', backref='student', lazy=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), default='Present')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('student_id', 'date', name='_student_date_uc'),)

# Create database tables
with app.app_context():
    db.create_all()
    # Create admin user if not exists
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(username='admin', email='admin@example.com', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        
        user = User.query.get(session['user_id'])
        if user.role != 'admin':
            flash('You do not have permission to access this page', 'danger')
            return redirect(url_for('dashboard'))
            
        return f(*args, **kwargs)
    return decorated_function

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_qr_code(data):
    qr = qc.QRCode(version=2, box_size=20, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save QR code
    filename = f"{data}.png"
    filepath = os.path.join('static/qrcodes', filename)
    img.save(filepath)
    
    return filepath

def generate_attendance_chart(student_id):
    student = Student.query.get(student_id)
    if not student:
        return None
    
    # Get attendance data
    attendances = Attendance.query.filter_by(student_id=student.id).all()
    total_days = len(attendances)
    present_days = sum(1 for a in attendances if a.status == 'Present')
    
    if total_days == 0:
        attendance_percentage = 0
    else:
        attendance_percentage = (present_days / total_days) * 100
    
    # Create pie chart
    plt.figure(figsize=(6, 6))
    plt.pie([attendance_percentage, 100 - attendance_percentage], 
            labels=['Present', 'Absent'], 
            autopct='%1.1f%%', 
            startangle=140,
            colors=['#4CAF50', '#F44336'])
    plt.title(f'Attendance Ratio for {student.first_name} {student.last_name}')
    
    # Save chart to memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Save chart to file
    chart_path = f'static/charts/{student.name_id}_attendance_chart.png'
    plt.savefig(chart_path)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    data = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{data}', chart_path

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not all([username, email, password, confirm_password]):
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    students_count = Student.query.count()
    attendance_today = Attendance.query.filter_by(date=datetime.utcnow().date()).count()
    
    # Get recent attendance records
    recent_attendance = db.session.query(Attendance, Student)\
        .join(Student)\
        .order_by(Attendance.created_at.desc())\
        .limit(5)\
        .all()
    
    return render_template('dashboard.html', 
                           students_count=students_count,
                           attendance_today=attendance_today,
                           recent_attendance=recent_attendance)

@app.route('/students')
@login_required
def students_list():
    students = Student.query.all()
    return render_template('students.html', students=students)

@app.route('/students/add', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        first_name = request.form.get('first_name').strip().title()
        last_name = request.form.get('last_name').strip().title()
        enrollment_no = request.form.get('enrollment_no').strip().upper()
        
        # Validate input
        if not all([first_name, last_name, enrollment_no]):
            flash('All fields are required', 'danger')
            return redirect(url_for('add_student'))
        
        # Check if enrollment number already exists
        if Student.query.filter_by(enrollment_no=enrollment_no).first():
            flash('Enrollment number already exists', 'danger')
            return redirect(url_for('add_student'))
        
        # Create name_id
        name_id = f"{first_name}{last_name}-{enrollment_no}"
        
        # Generate QR code
        qr_code_path = generate_qr_code(name_id)
        
        # Create new student
        new_student = Student(
            first_name=first_name,
            last_name=last_name,
            enrollment_no=enrollment_no,
            name_id=name_id,
            qr_code_path=qr_code_path
        )
        
        db.session.add(new_student)
        db.session.commit()
        
        flash('Student added successfully', 'success')
        return redirect(url_for('students_list'))
    
    return render_template('add_student.html')

@app.route('/students/<int:student_id>')
@login_required
def student_detail(student_id):
    student = Student.query.get_or_404(student_id)
    
    # Generate attendance chart
    chart_data, chart_path = generate_attendance_chart(student_id)
    
    # Get attendance records
    attendances = Attendance.query.filter_by(student_id=student_id).order_by(Attendance.date.desc()).all()
    
    return render_template('student_detail.html', 
                           student=student, 
                           chart_data=chart_data,
                           attendances=attendances)

@app.route('/students/<int:student_id>/delete', methods=['POST'])
@login_required
def delete_student(student_id):
    student = Student.query.get_or_404(student_id)
    
    # Delete attendance records
    Attendance.query.filter_by(student_id=student_id).delete()
    
    # Delete QR code file
    if student.qr_code_path and os.path.exists(student.qr_code_path):
        os.remove(student.qr_code_path)
    
    # Delete chart file
    chart_path = f'static/charts/{student.name_id}_attendance_chart.png'
    if os.path.exists(chart_path):
        os.remove(chart_path)
    
    # Delete student
    db.session.delete(student)
    db.session.commit()
    
    flash('Student deleted successfully', 'success')
    return redirect(url_for('students_list'))

@app.route('/attendance')
@login_required
def attendance_list():
    # Get date filter
    date_str = request.args.get('date')
    if date_str:
        try:
            filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            filter_date = datetime.utcnow().date()
    else:
        filter_date = datetime.utcnow().date()
    
    # Get attendance records for the selected date
    attendance_records = db.session.query(Attendance, Student)\
        .join(Student)\
        .filter(Attendance.date == filter_date)\
        .all()
    
    return render_template('attendance.html', 
                           attendance_records=attendance_records,
                           filter_date=filter_date)

@app.route('/mark-attendance')
@login_required
def mark_attendance_page():
    return render_template('mark_attendance.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    # Load face model
    model_path = 'static/models/face_recognition_model.h5'
    
    # If model doesn't exist, return error message
    if not os.path.exists(model_path):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Model not trained yet", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    face_model = load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    qr_code_detector = cv2.QRCodeDetector()
    
    # Get all students
    students = Student.query.all()
    student_dict = {student.name_id: student.id for student in students}
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Detect QR code
        qr_data, bbox, _ = qr_code_detector.detectAndDecode(frame)
        
        if qr_data and qr_data in student_dict:
            # Draw QR code bounding box
            if bbox is not None:
                bbox = bbox.astype(int)
                for i in range(len(bbox[0])):
                    cv2.line(frame, tuple(bbox[0][i]), tuple(bbox[0][(i+1) % len(bbox[0])]), color=(0, 255, 0), thickness=2)
            
            # Display QR code data
            cv2.putText(frame, f"QR: {qr_data}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Process faces
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (64, 64))
                face_roi = np.expand_dims(face_roi, axis=-1)
                face_roi = face_roi / 255.0
                face_roi = np.reshape(face_roi, (1, 64, 64, 1))
                
                # Make prediction
                prediction = face_model.predict(face_roi)
                predicted_label = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Display prediction results
                cv2.putText(frame, f"Conf: {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Mark attendance if QR code matches face
                if confidence > 70:  # Threshold for face recognition
                    student_id = student_dict.get(qr_data)
                    if student_id:
                        # Check if attendance already marked today
                        today = datetime.utcnow().date()
                        existing_attendance = Attendance.query.filter_by(
                            student_id=student_id, 
                            date=today
                        ).first()
                        
                        if not existing_attendance:
                            # Mark attendance
                            new_attendance = Attendance(
                                student_id=student_id,
                                date=today,
                                status='Present'
                            )
                            db.session.add(new_attendance)
                            db.session.commit()
                            
                            cv2.putText(frame, "Attendance Marked!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Already Marked", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/train-model')
@login_required
def train_model_page():
    return render_template('train_model.html')

@app.route('/start-training', methods=['POST'])
@login_required
def start_training():
    # Check if there are students registered
    students = Student.query.all()
    if not students:
        return jsonify({'status': 'error', 'message': 'No students registered yet'})
    
    # Create directories for training data
    os.makedirs('static/training_data', exist_ok=True)
    
    # Start training in a separate thread
    # For now, we'll just return success
    return jsonify({'status': 'success', 'message': 'Training started'})

@app.route('/train-model-status')
@login_required
def train_model_status():
    # Check if model exists
    model_path = 'static/models/face_recognition_model.h5'
    if os.path.exists(model_path):
        return jsonify({'status': 'completed', 'message': 'Model training completed'})
    else:
        return jsonify({'status': 'in_progress', 'message': 'Model training in progress'})

@app.route('/attendance-charts')
@login_required
def attendance_charts():
    students = Student.query.all()
    charts = []
    
    for student in students:
        # Generate chart if it doesn't exist
        chart_path = f'static/charts/{student.name_id}_attendance_chart.png'
        if not os.path.exists(chart_path):
            generate_attendance_chart(student.id)
        
        # Add chart to list
        charts.append({
            'student_id': student.id,
            'student_name': f"{student.first_name} {student.last_name}",
            'chart_path': chart_path
        })
    
    return render_template('attendance_charts.html', charts=charts)

@app.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)

@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    user = User.query.get(session['user_id'])
    
    username = request.form.get('username')
    email = request.form.get('email')
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    # Update username and email
    if username and username != user.username:
        # Check if username already exists
        if User.query.filter_by(username=username).first() and username != user.username:
            flash('Username already exists', 'danger')
            return redirect(url_for('profile'))
        user.username = username
        session['username'] = username
    
    if email and email != user.email:
        # Check if email already exists
        if User.query.filter_by(email=email).first() and email != user.email:
            flash('Email already exists', 'danger')
            return redirect(url_for('profile'))
        user.email = email
    
    # Update password
    if current_password and new_password and confirm_password:
        if not user.check_password(current_password):
            flash('Current password is incorrect', 'danger')
            return redirect(url_for('profile'))
        
        if new_password != confirm_password:
            flash('New passwords do not match', 'danger')
            return redirect(url_for('profile'))
        
        user.set_password(new_password)
    
    db.session.commit()
    flash('Profile updated successfully', 'success')
    return redirect(url_for('profile'))

@app.route('/users')
@admin_required
def users_list():
    users = User.query.all()
    return render_template('users.html', users=users)

@app.route('/users/add', methods=['GET', 'POST'])
@admin_required
def add_user():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        
        # Validate input
        if not all([username, email, password, role]):
            flash('All fields are required', 'danger')
            return redirect(url_for('add_user'))
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('add_user'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return redirect(url_for('add_user'))
        
        # Create new user
        new_user = User(username=username, email=email, role=role)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('User added successfully', 'success')
        return redirect(url_for('users_list'))
    
    return render_template('add_user.html')

@app.route('/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    # Prevent deleting own account
    if user_id == session['user_id']:
        flash('You cannot delete your own account', 'danger')
        return redirect(url_for('users_list'))
    
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    
    flash('User deleted successfully', 'success')
    return redirect(url_for('users_list'))

@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        role = request.form.get('role')
        password = request.form.get('password')
        
        # Update user details
        if username and username != user.username:
            # Check if username already exists
            if User.query.filter_by(username=username).first() and username != user.username:
                flash('Username already exists', 'danger')
                return redirect(url_for('edit_user', user_id=user_id))
            user.username = username
        
        if email and email != user.email:
            # Check if email already exists
            if User.query.filter_by(email=email).first() and email != user.email:
                flash('Email already exists', 'danger')
                return redirect(url_for('edit_user', user_id=user_id))
            user.email = email
        
        if role:
            user.role = role
        
        if password:
            user.set_password(password)
        
        db.session.commit()
        flash('User updated successfully', 'success')
        return redirect(url_for('users_list'))
    
    return render_template('edit_user.html', user=user)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)