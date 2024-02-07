# AI Work Assistant - Transcription

## Overview
This AI Work Assistant is designed to help professionals with their daily tasks, starting with a transcription function. It uses state-of-the-art ASR (Automatic Speech Recognition) and diarization models to transcribe audio input accurately.

## Installation

1. Clone this repository to your local machine.
2. Ensure you have Python 3.6+ installed.
3. Install the required dependencies by running `pip install -r requirements.txt` in the project's root directory.
4. Obtain a Hugging Face API token and set it in a `.env` file.

## Setup

### `.env` File
Create a `.env` file in the root directory of the project and add your Hugging Face API token as follows:

HF_KEY=your_hugging_face_api_token_here

### Running the Application

1. Start the application by running `python app.py` from the command line.
2. Open a web browser and navigate to `http://0.0.0.0:5000/` to access the web interface.
3. Use the web interface to start transcribing audio.

## Features

- Real-time audio capture and transcription.
- Dynamic adjustment of transcription sensitivity settings.
- Ability to select input devices for audio capture.

For more detailed instructions and troubleshooting, refer to the documentation inside the project.

## Contributing

Contributions to improve the AI Work Assistant are welcome. Please feel free to fork the repository, make improvements, and submit pull requests.

## License

TBD.


## Notes
- The application uses Flask's development server and is **not intended for production use**.
- **Ensure your microphone is set up and configured correctly** on your system.
- The application's **performance and accuracy depend** on the Whisper-Plus model and your hardware capabilities.

## Known Issue
sometimes the flask server is broken. flush socket pools to get it working in chrome: chrome://net-internals/#sockets
