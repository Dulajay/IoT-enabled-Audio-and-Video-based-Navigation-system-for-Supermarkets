import os
import subprocess
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

# Define the login credentials
login_credentials = {
    'expnav0001': '0000',
    'expnav0002': '0001'
}

# Variable to keep track of the subprocess
motor_process = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    login_id = request.form['login-id']
    password = request.form['password']

    if login_id in login_credentials and login_credentials[login_id] == password:
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/start_follow', methods=['POST'])
def start_follow():
    global motor_process
    try:
        # Start the motor process if it's not already running
        if motor_process is None:
            python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expnav', 'Scripts', 'python.exe')
            motor_process = subprocess.Popen([python_path, 'motor.py', 'follow'])
        else:
            motor_process.send_signal(subprocess.signal.SIGUSR1)  # Example signal
        return jsonify({'status': 'success', 'message': 'Motor script started for follow.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop():
    global motor_process
    try:
        if motor_process is not None:
            motor_process.send_signal(subprocess.signal.SIGUSR2)  # Example signal
        return jsonify({'status': 'success', 'message': 'Motor script stopped.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
