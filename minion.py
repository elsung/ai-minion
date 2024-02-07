from flask import Flask, render_template, jsonify, request
import pyaudio
import numpy as np
import wave
import json
import threading
from dotenv import load_dotenv
import os
import tempfile
import webbrowser
from whisperplus import ASRDiarizationPipeline, format_speech_to_dialogue
import collections
import time
import torch
import logging

# Load the variables from the .env file in the current directory
load_dotenv()

# Now you can use os.getenv to get your environment variable
token = os.getenv('HF_KEY')
if not token:
    raise ValueError("Token not found. Please set the HF_KEY for the huggingface token in the .env file.")

app = Flask(__name__)

# Configure Flask logger
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Set to ERROR to only log errors

# Updates Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('settings_update')
logger = logging.getLogger(__name__)

# Custom logger for status updates
status_logger = logging.getLogger('status_logger')
status_logger.setLevel(logging.INFO)

# Device configuration
device = "mps"  # Default to MPS
if not torch.backends.mps.is_available():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

pipeline = ASRDiarizationPipeline.from_pretrained(
    asr_model="openai/whisper-large-v3",
    diarizer_model="pyannote/speaker-diarization",
    use_auth_token=token,
    chunk_length_s=30,
    device=device,
)
transcribed_text = ""
running = True
status = "idle"

FRAMES_PER_BUFFER = 1024
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
MAX_SEGMENTS = 1000
RECORD_SECONDS = 1
SILENCE_THRESHOLD = 500
MINIMUM_AUDIBLE_COUNT = 1
PAUSE_THRESHOLD = 1000

# New constants for dynamic processing
MAX_ACCUMULATE_DURATION = 60  # Maximum duration to accumulate audio (in seconds)
MIN_ACCUMULATE_DURATION = 5   # Minimum duration to accumulate audio before processing (in seconds)
SIGNIFICANT_PAUSE_DURATION = 2.0  # Duration of pause to consider as a break in conversation

audio_capture_queue = collections.deque(maxlen=1000)

audio_segments = collections.deque(maxlen=MAX_SEGMENTS)
audible_count = 0
current_device_index = None

def get_default_device_index():
    p = pyaudio.PyAudio()
    default_device_index = p.get_default_input_device_info()['index']
    p.terminate()
    return default_device_index

current_device_index = get_default_device_index()

def calculate_volume_metric(audio_data):
    audio_as_int = np.frombuffer(audio_data, dtype=np.int16)
    return np.std(audio_as_int)

def save_audio_chunk(frames, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(AUDIO_FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

# Initialize the threads as None
capture_thread = None
transcribe_thread = None

def start_capture_thread():
    global capture_thread
    if capture_thread is not None and capture_thread.is_alive():
        capture_thread.join()
    capture_thread = threading.Thread(target=audio_capture_thread)
    capture_thread.start()

def start_transcription_thread():
    global transcribe_thread
    if transcribe_thread is not None and transcribe_thread.is_alive():
        transcribe_thread.join()
    transcribe_thread = threading.Thread(target=transcription_thread)
    transcribe_thread.start()

def audio_capture_thread():
    global running, audio_capture_queue, current_device_index

    p = pyaudio.PyAudio()
    device_index = current_device_index if current_device_index is not None else p.get_default_input_device_info()['index']
    stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER, input_device_index=device_index)

    try:
        while running:
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            audio_capture_queue.append(data)
            status_logger.debug("Captured audio data")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def transcription_thread():
    global transcribed_text, running, status, audio_capture_queue

    accumulated_frames = []
    last_process_time = time.time()

    while running:
        current_time = time.time()
        
        # Check if there's audio data to process
        while audio_capture_queue:
            # Update status to listening when capturing audio data
            if status != "listening":
                status = "listening"
                status_logger.info("Status changed to: Listening")
            
            accumulated_frames.append(audio_capture_queue.popleft())

        # Determine if it's time to process the accumulated audio
        if accumulated_frames and (current_time - last_process_time > MAX_ACCUMULATE_DURATION or len(accumulated_frames) * FRAMES_PER_BUFFER / SAMPLE_RATE >= MIN_ACCUMULATE_DURATION):
            status = "transcribing"
            status_logger.info("Status changed to: Transcribing")
            process_accumulated_audio(accumulated_frames)
            
            # Clear the accumulated frames after processing
            accumulated_frames = []
            last_process_time = current_time

            # After processing, if there's no more audio data, set status to idle
            if not audio_capture_queue:
                status = "idle"
                status_logger.info("Status changed to: Idle")

        # If there are no frames to process, ensure status is set to idle
        elif not accumulated_frames and status != "idle":
            status = "idle"
            status_logger.info("Status changed to: Idle")

        time.sleep(0.01)  # Prevent excessive CPU usage

def process_accumulated_audio(frames):
    global transcribed_text
    volume_metric = calculate_volume_metric(b''.join(frames))
    
    if volume_metric < SILENCE_THRESHOLD:
        print("Skipping low volume audio segment.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
        save_audio_chunk(frames, temp_audio_file.name)

    temp_audio_file.close()
    try:
        output_text = pipeline(temp_audio_file.name, num_speakers=2, min_speaker=1, max_speaker=2)
        dialogue = format_speech_to_dialogue(output_text)
        transcribed_text += dialogue
        print(dialogue)
    except IndexError:
        print("No speech segments detected in the audio.")
    except ValueError as e:
        print(f"Error processing accumulated audio: {e}")

    with open("transcription.json", "w") as file:
        json.dump({"transcription": transcribed_text}, file)

    try:
        os.remove(temp_audio_file.name)
    except PermissionError as e:
        print(f"Error deleting file {temp_audio_file.name}: {e}")

@app.route('/get_devices')
def get_devices():
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            devices.append({'index': i, 'name': dev['name']})
    p.terminate()
    return jsonify(devices)

@app.route('/set_device', methods=['POST'])
def set_device():
    global current_device_index
    device_index = request.json.get('device_index')
    if device_index is not None:
        current_device_index = int(device_index)
        print(f"Switching to device index: {current_device_index}")
        start_capture_thread()  # Restart the capture thread with the new device
        return jsonify({'status': 'success', 'message': 'Device set successfully'})
    return jsonify({'status': 'error', 'message': 'Invalid device index'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_transcription')
def get_transcription():
    return jsonify({"transcription": transcribed_text})

@app.route('/status')
def get_status():
    return jsonify({"status": status})

@app.route('/get_current_device')
def get_current_device():
    global current_device_index
    return jsonify({'current_device_index': current_device_index})

@app.route('/update_settings', methods=['POST'])

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global MAX_SEGMENTS, RECORD_SECONDS, SILENCE_THRESHOLD, MINIMUM_AUDIBLE_COUNT, PAUSE_THRESHOLD, MAX_ACCUMULATE_DURATION, MIN_ACCUMULATE_DURATION, SIGNIFICANT_PAUSE_DURATION, audio_capture_queue

    settings = request.json

    try:
        if 'max_segments' in settings:
            MAX_SEGMENTS = int(settings['max_segments'])
        if 'record_seconds' in settings:
            RECORD_SECONDS = int(settings['record_seconds'])
        if 'silence_threshold' in settings:
            SILENCE_THRESHOLD = int(settings['silence_threshold'])
        if 'minimum_audible_count' in settings:
            MINIMUM_AUDIBLE_COUNT = int(settings['minimum_audible_count'])
        if 'pause_threshold' in settings:
            PAUSE_THRESHOLD = int(settings['pause_threshold'])
        if 'max_accumulate_duration' in settings:
            MAX_ACCUMULATE_DURATION = int(settings['max_accumulate_duration'])
        if 'min_accumulate_duration' in settings:
            MIN_ACCUMULATE_DURATION = int(settings['min_accumulate_duration'])
        if 'significant_pause_duration' in settings:
            SIGNIFICANT_PAUSE_DURATION = float(settings['significant_pause_duration'])
        
        # Adjusting the maxlen of the deque according to the updated settings if needed
        audio_capture_queue = collections.deque(maxlen=MAX_SEGMENTS)

        logger.info("Settings updated: %s", settings)  # Log the updated settings
        return jsonify({'status': 'success', 'message': 'Settings updated successfully'})
    except Exception as e:
        logger.error("Failed to update settings: %s", e)
        return jsonify({'status': 'error', 'message': 'Failed to update settings: ' + str(e)})


@app.route('/quit', methods=['POST'])
def quit():
    shutdown_function = request.environ.get('werkzeug.server.shutdown')
    if shutdown_function is not None:
        try:
            shutdown_function()
            logger.info('Server shutting down...')
            return jsonify({"status": "success", "message": "Server is shutting down"})
        except Exception as e:
            logger.error(f'Graceful shutdown failed: {e}')
    else:
        logger.error('Shutdown not possible through Werkzeug. Attempting forceful shutdown.')

    # Forceful shutdown as a last resort
    try:
        os._exit(0)
    except Exception as e:
        logger.error(f'Forceful shutdown failed: {e}')
        return jsonify({"status": "error", "message": "Forceful shutdown failed"})
    return jsonify({"status": "success", "message": "Server is forcefully shutting down"})


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG for detailed output
    start_capture_thread()
    start_transcription_thread()
    webbrowser.open("http://0.0.0.0:5000/")
    app.run(debug=True, use_reloader=False)
