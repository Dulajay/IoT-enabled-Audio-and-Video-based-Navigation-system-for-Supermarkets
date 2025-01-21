import os
import subprocess
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

# Define the login credentials
login_credentials = {
    'expnav0001': '0000',
    'expnav0002': '0001'
}

# Variable to keep track of the subprocess
motor_process = None

# Store logged-in user information and motor status globally
logged_in_user = None
login_time = None
motor_status = "Idle"  # Initially, the robot is not following

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    global logged_in_user, login_time
    login_id = request.form['login-id']
    password = request.form['password']

    if login_id in login_credentials and login_credentials[login_id] == password:
        logged_in_user = login_id  # Save the logged-in user
        login_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Capture login time
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/start_follow', methods=['POST'])
def start_follow():
    global motor_process, motor_status
    motor_status = "Following"
    try:
        # Start the motor process if it's not already running
        if motor_process is None:
            python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expnav', 'Scripts', 'python.exe')
            motor_process = subprocess.Popen([python_path, 'motor.py', 'follow'])
        return jsonify({'status': 'success', 'message': 'Motor script started for follow.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop():
    global motor_process, motor_status
    motor_status = "Stopped"
    try:
        if motor_process is not None:
            motor_process.terminate()  # Terminate the process
            motor_process = None  # Reset the motor process
        return jsonify({'status': 'success', 'message': 'Motor script stopped.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Monitoring route to provide current status
@app.route('/monitor')
def monitor():
    global logged_in_user, login_time, motor_status
    return jsonify({
        'user': logged_in_user,
        'login_time': login_time,
        'motor_status': motor_status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Listen on all available interfaces
