import os
import cv2
import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Configuration
clips_dir = "dataset/10_dragonball"
metadata_file = "dataset/10_dragonball/metadata.json"
output_metadata = "dataset/10_dragonball/updated_metadata.json"
frames_per_clip = 5  # One frame per second for a 5-second clip

gpu_index = int(os.environ.get("BLIP2_GPU", 0))
# Load BLIP-2 model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")
print("script started")
# Load existing metadata
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Generate summaries
new_metadata = []
for item in metadata:
    video_path = os.path.join(clips_dir, item['file_name'])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps) if fps > 0 else 24  # Default to 24 FPS if unavailable
    frames = []
    
    for i in range(frames_per_clip):
        frame_time = (i * frame_interval) // fps  # Approximate second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_time * fps)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            break
    
    cap.release()
    
    if not frames:
        print(f"Failed to read frames from {video_path}")
        continue
    
    # Process all frames with BLIP-2
    inputs = processor(images=frames, return_tensors="pt").to("cuda", torch.float16)
    generated_ids = model.generate(**inputs)
    captions = [processor.decode(gid, skip_special_tokens=True) for gid in generated_ids]
    
    # Combine captions into a concise summary
    summary_parts = [cap for cap in captions if cap]  # Filter empty captions
    summary = ". ".join(summary_parts[:3])  # Limit to 3 sentences for brevity
    if summary:
        summary += " in a Dragon Ball Z style fight"
    else:
        summary = "Action scene in a Dragon Ball Z style fight"  # THis fall back is important make sure you know what you are doing bisrat. 
    
    new_metadata.append({"file_name": item['file_name'], "caption": summary})

# Save updated metadata
with open(output_metadata, 'w') as f:
    json.dump(new_metadata, f, indent=2)

print(f"Summaries saved to {output_metadata}")