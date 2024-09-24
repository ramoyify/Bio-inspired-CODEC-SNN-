import os
import cv2
from moviepy.editor import VideoFileClip

# File path for the original video
original_video_path = '/home/ntu-user/PycharmProjects/Assessment/Video/COMP40731_video.mp4'
# Directory to save frames and sound
frames_dir = 'Frames'
sound_dir = 'Sound'
video_chunks_dir = 'VideoChunks'
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(sound_dir, exist_ok=True)
os.makedirs(video_chunks_dir, exist_ok=True)

# Load the original video clip
video_clip = VideoFileClip(original_video_path)

# Define duration and frame rate
duration = video_clip.duration
frame_rate = video_clip.fps

def save_audio_chunks_as_wav(video_clip, output_dir):
    chunk_duration = 1 / frame_rate  # Calculate chunk duration based on frame rate
    num_chunks = int(duration / chunk_duration)
    print(f"Total Duration: {duration} seconds")
    print(f"Frame Rate: {frame_rate} frames per second")
    print(f"Chunk Duration: {chunk_duration} seconds")
    print(f"Total Chunks: {num_chunks}")
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, duration)
        chunk_audio = video_clip.subclip(start_time, end_time).audio
        output_path = os.path.join(output_dir, f"chunk_{i+1}.wav")
        print(f"Saving chunk {i+1} to {output_path}")
        chunk_audio.write_audiofile(output_path, codec='pcm_s16le', fps=chunk_audio.fps)

    return num_chunks, chunk_duration

# Save the audio as chunks
num_chunks, chunk_duration = save_audio_chunks_as_wav(video_clip, video_chunks_dir)

# Save the audio
audio_filename = os.path.join(sound_dir, 'extracted_audio.wav')
video_clip.audio.write_audiofile(audio_filename, codec='pcm_s16le', fps=video_clip.audio.fps)
print(f"Audio saved to: {audio_filename}")

# Save each frame as a separate JPEG image in the Frames folder
for frame_number, frame in enumerate(video_clip.iter_frames(), start=1):
    # Save the frame as a JPEG image
    frame_filename = os.path.join(frames_dir, f"frame{frame_number:04d}.jpg")
    cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# If there's an extra frame, save it as a separate audio chunk
if len(os.listdir(frames_dir)) > num_chunks:
    start_time = num_chunks * chunk_duration
    end_time = duration
    chunk_audio = video_clip.subclip(start_time, end_time).audio
    output_path = os.path.join(video_chunks_dir, f"chunk_{num_chunks+1}.wav")
    print(f"Saving extra chunk {num_chunks+1} to {output_path}")
    chunk_audio.write_audiofile(output_path, codec='pcm_s16le', fps=chunk_audio.fps)

# Close the video clip
video_clip.close()

print("Frames and audio chunks saved successfully.")
