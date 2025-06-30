import os
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import json
from fm_solvers import FlowDPMSolverMultistepScheduler

# Custom Dataset for Video Clips
class DragonBallVideoDataset(Dataset):
    def __init__(self, data_dir, metadata_file):
        self.data_dir = data_dir
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((480, 854))  # Match Wan 2.1 T2V-1.3B input (480p)
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.metadata[idx]['file_name'])
        caption = self.metadata[idx]['caption']
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame_rgb)
            frames.append(frame_tensor)
        cap.release()
        return {"frames": torch.stack(frames), "caption": caption}

# Load Model, VAE, and Configure LoRA
base_model = "./Wan2.1-T2V-1.3B"
vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float16).to("cuda")
pipeline = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    temporal_attention=True
)
pipeline.unet = get_peft_model(pipeline.unet, lora_config)
lora_layers = filter(lambda p: p.requires_grad, pipeline.unet.parameters())

# Text Encoder
text_encoder = pipeline.text_encoder.to("cuda")

# Scheduler
scheduler = FlowDPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    solver_order=2,
    prediction_type="flow_prediction",
    shift=1.0,
    use_dynamic_shifting=False
)
scheduler.set_timesteps(num_inference_steps=50, device="cuda")  # Adjust steps as needed

# Dataset and Dataloader
dataset = DragonBallVideoDataset(
    data_dir="dataset/10_dragonball",
    metadata_file="dataset/10_dragonball/metadata.json"
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Loss Computation Function
def compute_loss(model, vae, scheduler, x_0, t, noise, text_embeds, device="cuda"):
    with torch.no_grad():
        x_0_latent = vae.encode(x_0).latent_dist.sample()

    def get_timestep_embedding(timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = torch.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    noise_scale = get_timestep_embedding(t, x_0_latent.shape[1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    x_t = x_0_latent + noise * noise_scale

    pred_noise = model(x_t, t, text_embeds).sample
    noise_loss = torch.nn.functional.mse_loss(pred_noise, noise, reduction='mean')

    # Flow Matching Loss
    sigma_t = scheduler.sigmas[scheduler.step_index].to(device)
    model_output = model(x_t, t, text_embeds)
    flow_pred = scheduler.convert_model_output(model_output, sample=x_t)
    v_star = noise / sigma_t  # Simplified optimal vector field (placeholder; adjust based on Wan specifics), How can I get this, dont forget to change it 
    flow_loss = torch.nn.functional.mse_loss(flow_pred, v_star, reduction='mean')

    total_loss = noise_loss + 0.1 * flow_loss  # Weight flow loss (adjust as needed)
    return total_loss

# Training Loop
optimizer = torch.optim.AdamW(lora_layers, lr=1e-4)
num_epochs = 2 #experiment on this one 
for epoch in range(num_epochs):
    for batch in dataloader:
        frames, captions = batch["frames"].to("cuda"), batch["caption"]
        t = torch.randint(0, 1000, (frames.shape[0],), device="cuda").long()
        noise = torch.randn_like(frames)
        text_embeds = text_encoder(captions)[0]

        optimizer.zero_grad()
        loss = compute_loss(pipeline.unet, vae, scheduler, frames, t, noise, text_embeds)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs} completed, Loss: {loss.item():.4f}")

# Save LoRA Weights
output_dir = "./output/dragonball_lora"
os.makedirs(output_dir, exist_ok=True)
pipeline.save_pretrained(output_dir)
print(f"LoRA weights saved to {output_dir}/dragonball_lora.safetensors")