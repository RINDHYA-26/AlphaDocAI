whisper_model = load_whisper_model()
audio_bytes = audio_file.read()
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
    temp_audio.write(audio_bytes)
    temp_audio_path = temp_audio.name

segments, info = whisper_model.transcribe(
    temp_audio_path,
    beam_size=5,
    temperature=0.0,
    initial_prompt=(...)
)
texts = [seg.text if hasattr(seg, "text") else seg[2] for seg in segments]
transcribed_text = " ".join(texts).strip()
