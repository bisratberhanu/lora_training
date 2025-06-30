import os # for operating system related purposes 
import torch
from  diffusers import DiffusionPipeline  
from peft import LoraConfig, get_peft_model 
from torch.utils.data import DataSet, DataLoader 
from torchvison import transforms # for resizing purposes 
import cv2
import json # for parsing meta data 


class DragonBallDataSet(DataSet):
    def __init__(self, data_dir, metadata_file):
        self.data_dir = data_dir
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
        
        self.transform  = transforms.Compose([transforms.toTensor(),transforms.Resize(480,854)]) # match wan 2.1 output 
    
    def __len__(self):
        return len(self.metadata)
    
    def __get_item__(self, idx):
        video_path = os.path.join(self.data_dir, self.metadata[idx]["file_name"])
        caption = self.metadata["caption"]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.BGR2RGB)
            frame_tensor = self.transform(frame_rgb)
            frames.append(frame_tensor)
        
        cap.release()
        return {"frames": torch.stack(frames),"caption": caption}


#LORA: low rank adapation for training large models by adding trainable low rank matrics 
base_model = "./wan2.1-T2V-1.3B" # make sure this path is correct
pipeline = DiffusionPipeline.from_pretrained(base_model,torch_dtype = torch.float16).to("cuda")
lora_config = LoraConfig(
    r= 64, #rank
    lora_alpha = 32,
    init_lora_weights = "guassian",
    target_modules = ["to_k","to_q", "to_v", "to_out.0"],
    temporal_attention = True
)

pipeline.unet = get_peft_model(pipeline.unet, lora_config)
lora_layers = filter(lambda p: p.requires_grad, pipeline.unet.parameters())

dataset = DragonBallDataSet(
    data_dir= "dataset/10_dragonball",
    metadata_file="dataset/10_dragonball/metadata.json"
)

dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

optimizer = torch.optim.AdamW(lora_layers, lr = 1e-4)
num_epochs = 2 #make sure to channge this to the optimum value 

for epoch in range(num_epochs):
    for batch in dataloader:
        frames, captions = batch["frames"].to("cuda"), batch["caption"]
        optimizer.zero_grad()
        loss = pipeline.unet(frames, captions) #this line needs study since things are not this easy in wan models, diffusion models use noise based loss functions 
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1 } / {num_epochs} completed")


pipeline.save_pretrained("./output/dragonball_lora") #gives a safe tensor file 



