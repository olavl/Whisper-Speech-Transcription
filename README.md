# Whisper-Speech-Transcription
A simple desktop application for speech-to-text conversion using OpenAI's Whisper model.

## Description
Whisper Speech Transcription Tool is a standalone desktop application designed for real-time speech-to-text conversion using a local copy of OpenAI's Whisper large V3 model. While it performs optimally on systems with robust hardware, it's built to be self-contained, requiring no external APIs. Suitable for various OS platforms. 

<img src="https://github.com/olavl/Whisper-Speech-Transcription/blob/main/screenshot.png" width="485" height="350">


## Computational Requirements
- A robust CPU or CUDA-compatible GPU for optimal performance.
- Python 3.x installed on the system.

## Setup and Installation

### Python Interpreter
Ensure Python 3.8-3.11 is installed on your system. Download from [Python's official website](https://www.python.org/downloads/).

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
pip install accelerate huggingface-hub numpy PyAudio soundfile tokenizers transformers

```

## Usage
Run the provided batch file (or create equivalent in macOS/Linux) to activatet the virtual environment and launch the application.

## FAQs
- **Requirement for a robust computer?**
  - The application requires a powerful CPU or GPU.
  - NVIDIA GeForce RTX 4070 is more than enough.
  - Running it on CPU causes 100% processor capacity for a few seconds.
- **Alternatives to this tool?**
  - Browser-based solutions or other desktop applications might offer similar functionalities but often require more setup or API access.

## License
This project is distributed under the MIT License.

## Acknowledgments

Built with help of Grimoire, Coding Wizard GPT by Nick Dobos @NickADobos https://mindgoblinstudios.com/

