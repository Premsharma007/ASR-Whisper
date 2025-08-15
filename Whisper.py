import gradio as gr
import whisper
import os
import datetime
import torch
import time  # For timing

# -----------------------
# Config
# -----------------------
MODEL_SIZE = "large-v3"
LANGUAGE = "ta"
OUTPUT_DIR = "Transcriptions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading Whisper model: {MODEL_SIZE}...")

if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    fp16_mode = True
else:
    device = "cpu"
    print("CUDA not available. Using CPU. Transcription will be significantly slower.")
    fp16_mode = False

model = whisper.load_model(MODEL_SIZE, device=device)
print("Whisper model loaded successfully.")

# -----------------------
# Transcription function
# -----------------------
def transcribe_audio(audio_file_path):
    if audio_file_path is None:
        return "Please upload an audio file.", None, ""

    try:
        start_time = time.time()
        print(f"Starting transcription for: {audio_file_path}")

        result = model.transcribe(audio_file_path, language=LANGUAGE, fp16=fp16_mode)
        transcribed_text = result["text"]

        total_time = time.time() - start_time
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(f"Transcription complete in {total_time_str}.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.basename(audio_file_path)
        base_name = os.path.splitext(original_filename)[0]
        output_txt_filename = f"{base_name}_{timestamp}.txt"
        output_txt_path = os.path.join(OUTPUT_DIR, output_txt_filename)

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)

        print(f"Transcription saved to: {output_txt_path}")

        return transcribed_text, output_txt_path, total_time_str

    except Exception as e:
        error_message = f"An error occurred during transcription: {e}\n\n" \
                        f"Please ensure FFmpeg is installed and correctly configured in your system's PATH.\n" \
                        f"If using GPU, check your CUDA/cuDNN installation and PyTorch's CUDA availability."
        print(error_message)
        return error_message, None, ""

# -----------------------
# Gradio UI
# -----------------------
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Upload Tamil Audio (MP3, WAV, etc.)"),
    outputs=[
        gr.Textbox(label="Transcribed Text", lines=10, interactive=False),
        gr.File(label="Download Transcription (.txt)"),
        gr.Label(label="Total Time Taken (HH:MM:SS)")
    ],
    title="Spider『X』ASR Whisper Transcription Model - 1.0",
    description="Upload audio file (around 20 mins) to get its precise transcription. Total time taken will be shown below.",
    live=False,
    allow_flagging="auto",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    iface.launch(inbrowser=False, show_error=True)
