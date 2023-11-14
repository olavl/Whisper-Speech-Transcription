# Whisper-Speech-Transcription
A simple desktop application for speech-to-text conversion using OpenAI's Whisper model.

## Description
Whisper Speech Transcription Tool is a standalone desktop application designed for real-time speech-to-text conversion using OpenAI's Whisper model. While it performs optimally on systems with robust hardware, it's built to be self-contained, requiring no external APIs. Suitable for various OS platforms, with specific setup instructions provided for each.

## Computational Requirements
- A robust CPU or CUDA-compatible GPU for optimal performance.
- Python 3.x installed on the system.

## Setup and Installation

### Python Interpreter
Ensure Python is installed on your system. Download from [Python's official website](https://www.python.org/downloads/).

### Setting up the Virtual Environment
1. Install virtualenv:
   ```bash
   pip install virtualenv
   ```
2. Navigate to the project directory and create a virtual environment:
   ```bash
   virtualenv stt_env
   ```
3. Activate the virtual environment (Windows):
   ```bash
   .\stt_env\Scripts\activate
   ```

### Custom PyTorch Installation
Install PyTorch according to your system's capabilities. [Pytorch's offical website](https://pytorch.org/)
For example: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### Additional Dependencies
Install other required packages:
```bash
pip install transformers pyaudio wave numpy soundfile
```

## Usage
Run the provided batch file (or its equivalent in macOS/Linux) to set up the environment and launch the application.

## FAQs
- **Requirement for a robust computer?**
  - Optimally, yes. The application requires a powerful CPU or GPU.
- **Alternatives to this tool?**
  - Browser-based solutions or other desktop applications might offer similar functionalities but often require more setup or API access.

## License
This project is distributed under the MIT License.
