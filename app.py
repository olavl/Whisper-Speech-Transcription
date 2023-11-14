import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import wave
import os
import tkinter as tk
from tkinter import messagebox  
import numpy as np
import soundfile as sf
import threading

# Determine device based on CUDA availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Initialize the Whisper model and processor
def initialize_whisper(model_id="openai/whisper-large-v3"):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )


import threading
import pyaudio
import wave

class AudioRecorder:
    def __init__(self, output_filename="temp_audio.wav"):
        self.audio_format = pyaudio.paInt16  # Audio format (16-bit PCM)
        self.channels = 1                    # Number of audio channels (1 for mono)
        self.rate = 44100                    # Sampling rate (44.1kHz)
        self.chunk = 1024                    # Frames per buffer
        self.output_filename = output_filename
        self.frames = []                     # Container for frame data
        self.is_recording = False            # Recording state flag
        self.audio = pyaudio.PyAudio()       # PyAudio instance
        self.stream = self.audio.open(format=self.audio_format, channels=self.channels,
                                      rate=self.rate, input=True, frames_per_buffer=self.chunk)

    def start_recording(self):
        """Start the recording process in a separate thread."""
        if not self.is_recording:
            self.frames.clear()  # Clear any previous recording data
            self.is_recording = True
            threading.Thread(target=self.record).start()
            
            
    def stop_recording(self):
        """Stop the recording process."""
        self.is_recording = False

    def record(self):
        """Record audio from the default input device."""
        while self.is_recording:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            self.frames.append(data)
            
    def save_recording(self):
        """Save the recorded audio to a file."""
        wf = wave.open(self.output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def close_stream(self):
        """Close the audio stream and terminate PyAudio instance."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

def convert_speech_to_text(audio_file, whisper_pipeline):
    try:
        audio_data, sampling_rate = sf.read(audio_file, dtype='float32')
        input_dict = {"raw": audio_data, "sampling_rate": sampling_rate}
        result = whisper_pipeline(input_dict)
        return result.get("text", "")
    except Exception as e:
        print(f"Error during conversion: {e}")
        return ""

def create_app_window(whisper_pipeline):
    app = tk.Tk()
    app.title("Whisper Speech Transcription")

    recorder = AudioRecorder()
    text_box = tk.Text(app, wrap=tk.WORD)
    text_box.grid(row=0, column=0, columnspan=3, sticky="nsew")  # Span across 3 columns

    # Configure grid layout
    app.grid_rowconfigure(0, weight=1)
    for i in range(3):
        app.grid_columnconfigure(i, weight=1)  # Configure 3 columns

    record_btn_text = tk.StringVar()
    record_btn_text.set("Record")

    def on_app_close():
        recorder.close_stream()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_app_close)

    def toggle_recording():
        if recorder.is_recording:
            recorder.stop_recording()
            record_btn_text.set("Record")
            record_btn['state'] = tk.DISABLED  # Disable the record button
            app.after(1000, lambda: process_audio(recorder.output_filename))
        else:
            recorder.start_recording()
            record_btn_text.set("Recording...")

    def process_audio(audio_file):
        recorder.stop_recording()  # Ensure recording is stopped
        recorder.save_recording()  # Save the new recording

        transcribed_text = convert_speech_to_text(audio_file, whisper_pipeline)
        current_text = text_box.get(1.0, tk.END)
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, current_text + transcribed_text)
        record_btn['state'] = tk.NORMAL  # Re-enable the record button

    def clear_text_box():
        text_box.delete(1.0, tk.END)

    def copy_to_clipboard():
        app.clipboard_clear()
        app.clipboard_append(text_box.get(1.0, tk.END))

    record_btn = tk.Button(app, textvariable=record_btn_text, command=toggle_recording)
    record_btn.grid(row=1, column=0, pady=10)

    clear_btn = tk.Button(app, text="Clear Text", command=clear_text_box)
    clear_btn.grid(row=1, column=1, pady=10)

    copy_btn = tk.Button(app, text="Copy to Clipboard", command=copy_to_clipboard)
    copy_btn.grid(row=1, column=2, pady=10)

    return app

def main():
    whisper_pipeline = initialize_whisper()
    app = create_app_window(whisper_pipeline)
    app.mainloop()

if __name__ == "__main__":
    main()