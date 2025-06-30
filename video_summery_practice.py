import os
import cv2 # library for computer vision
import json 
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

#configuration
clips_dir = "dataset/10_dragonball"
metadata_file = "dataset/10_dragonball/metadata.json" # make sure the file exists
output_metadata = "dataset/10_dragonball/updated_metadata.json"
frames_per_clip = 5

#Loading the model
processor = Blip2Processor.from_pretrained("salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("salesforce/blip2-opt-2.7b", torch_dtype = torch.float16.to("cuda"))
#what is the difference between processor and model: processor #mean basically a preprocessor for both image and also text 


# Load existing metadata
with open(metadata_file, "r") as f:
    metadata = json.load(f)

new_metadata = []
for item in metadata:
    video_path = os.path.join(clips_dir, item["file_name"])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROPS_FPS) # to get the frame persecond
    frame_interval = int(fps) if fps > 0 else 24
    frames = []
    
    for i in range(frames_per_clip):
        frame_time = (i * frame_interval) // fps
        cap.set(cv2.CAP_PROPS_POS_FRAMES, frame_time*fps) #what is this line doing 
        ret,frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtcolor(frame,cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            break
    
    cap.release()

    if not frame:
        print("no frames was found")
        continue
    #now lets process all the frames using blip model
    inputs = processor(images = frames, return_tensors = "pt").to("cuda", torch.float16)
    generated_ids = model.generate(**inputs)
    captions = [processor.decode(gid, skip_special_tokens=True) for gid in generated_ids] # to human readable format

    #comibining to get a concise summary
    summary_parts = [cap for cap in captions if cap]
    summary = ".".join(summary_parts[:3])
    if summary:
        summary += "dragon ball anime"
    else:
        summary = "dragon ball z anime "
    
    
    new_metadata.append({"file_name": item['file_name'], "caption": summary})



# Save updated metadata
with open(output_metadata, 'w') as f:
    json.dump(new_metadata, f, indent=2)

print(f"Summaries saved to {output_metadata}")


# research if there is another approach 
