import gradio as gr
import whisper
import os
import datetime
import torch # Import torch to explicitly check for CUDA

# Define the model size
MODEL_SIZE = "large-v3"
# Tamil language code for Whisper
LANGUAGE = "ta"
# Output directory for transcripts
OUTPUT_DIR = "Transcriptions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the Whisper model once when the script starts
print(f"Loading Whisper model: {MODEL_SIZE}...")

# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    # For 4GB VRAM, medium model with fp16=True might just fit for longer audios
    # but fp16=False is safer initially. You can try True if OOM.
    fp16_mode = True # Set to True for potential memory savings on GPUpip install --upgrade gradio
else:
    device = "cpu"
    print("CUDA not available. Using CPU. Transcription will be significantly slower.")
    fp16_mode = False # fp16 has no benefit on CPU

model = whisper.load_model(MODEL_SIZE, device=device)
print("Whisper model loaded successfully.")

def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file using the pre-loaded Whisper model.
    Handles potential audio format issues and saves the output to a text file.
    """
    if audio_file_path is None:
        return "Please upload an audio file.", None

    try:
        print(f"Starting transcription for: {audio_file_path}")
        
        # Transcribe the audio
        # Using fp16_mode determined at startup
        result = model.transcribe(audio_file_path, language=LANGUAGE, fp16=fp16_mode)

        transcribed_text = result["text"]
        print("Transcription complete.")

        # Generate a unique filename for the text output
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.basename(audio_file_path)
        # Clean filename to remove invalid characters if any, though os.path.splitext handles most
        base_name = os.path.splitext(original_filename)[0]
        output_txt_filename = f"{base_name}_{timestamp}.txt"
        output_txt_path = os.path.join(OUTPUT_DIR, output_txt_filename)

        # Save the transcription to a .txt file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
        print(f"Transcription saved to: {output_txt_path}")

        # Return the transcription and the path to the saved file for download
        return transcribed_text, output_txt_path

    except Exception as e:
        error_message = f"An error occurred during transcription: {e}\n\n" \
                        f"Please ensure FFmpeg is installed and correctly configured in your system's PATH.\n" \
                        f"If using GPU, check your CUDA/cuDNN installation and PyTorch's CUDA availability."
        print(error_message)
        return error_message, None

# Create the Gradio interface
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Upload Tamil Audio (MP3, WAV, etc.)"),
    outputs=[
        gr.Textbox(label="Transcribed Text", lines=10, interactive=False),
        gr.File(label="Download Transcription (.txt)")
    ],
    title="Spider『X』ASR Whisper Transcription Model - 1.0",
    description=f"Upload audio file (around 20 mins) to get its precise Transcription.",
    live=False,
    allow_flagging="auto", # Allows users to flag incorrect outputs
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    # Launch the interface
    # share=True creates a public link (useful for sharing demos, but it's temporary and not secure for sensitive data)
    # For local use, share=False is generally preferred.

    iface.launch(inbrowser=False, show_error=True) # Opens in browser automatically, shows detailed errors


