import mlx_whisper
from pydub import AudioSegment
from pathlib import Path


# return output file
def clip(file_path: str, start_seconds: float, stop_seconds: float) -> str:
    input_path = Path(file_path)

    # check whether file exsits or not
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # check whether start(stop) time correct or not
    if start_seconds < 0:
        raise ValueError("start_seconds cannot be negative")

    if stop_seconds <= start_seconds:
        raise ValueError("stop_seconds must be greater than start_seconds")

    # load audio
    audio = AudioSegment.from_file(input_path)

    duration_seconds = len(audio) / 1000

    if stop_seconds > duration_seconds:
        raise ValueError(
            f"stop_seconds ({stop_seconds}) exceeds audio duration "
            f"({duration_seconds:.2f} seconds)"
        )
    # convert s to ms
    start_ms = int(start_seconds * 1000)
    stop_ms = int(stop_seconds * 1000)

    # clip audio
    clipped_audio = audio[start_ms:stop_ms]

    output_path = input_path.with_name(
        f"{input_path.stem}_{start_seconds:g}_{stop_seconds:g}.wav"
    )
    clipped_audio.export(output_path, format="wav")
    return str(output_path)


# test
output_file = clip("harvard.wav", 3, 5)
print(f"Created clip: {output_file}")


# also requires this package: audioop-lts


result = mlx_whisper.transcribe(
    output_file,
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    word_timestamps=True,
)

print(result["text"])
