import subprocess
from pathlib import Path
import sys

videos_path = Path("videos")
output_path = Path("outputs")
output_path.mkdir(exist_ok=True)
vid_list = sorted(videos_path.glob("*.mp4"))

for input in vid_list:
    output = output_path / f"{input.stem}_result.mp4"

    print(f"Processing: {input.name}")

    subprocess.run([
        sys.executable,
        "inputs/detect.py",
        str(input),
        str(output)
    ], check=True)

print("\n All Videos are processed")