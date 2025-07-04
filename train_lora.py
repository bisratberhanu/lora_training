import os
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import json
from diffusers import UniPCMultistepScheduler

# Custom Dataset for Video Clips
class DragonBallVideoDataset(Dataset):
    def __init__(self, data_dir, metadata_file):
        self.data_dir = data_dir
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((480, 854))  # Match Wan 2.1 I2V-14B input (480p)
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

gpu_index = int(os.environ.get("LORA_GPU", 0))
#make sure to run the command  LORA_GPU=1 python train_lora.py
device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Load Model, VAE, and Configure LoRA
base_model = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
pipeline = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
pipeline.to(device)
vae = pipeline.vae  # Use the VAE from the pipeline

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
text_encoder = pipeline.text_encoder.to(device)

# Scheduler
scheduler = UniPCMultistepScheduler(
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    num_train_timesteps=1000,
    solver_order=2,
    solver_type="bh2",
    prediction_type="flow_prediction",
    flow_shift=3.0,
    timestep_spacing="linspace",
    use_flow_sigmas=True
)
scheduler.set_timesteps(num_inference_steps=50, device="cuda")  # Adjust steps as needed

# Dataset and Dataloader
dataset = DragonBallVideoDataset(
    data_dir="dataset/10_dragonball",
    metadata_file="dataset/10_dragonball/metadata.json"
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True) #ram eater line 

# Loss Computation Function
def compute_loss(model, vae, scheduler, x_0, t, noise, text_embeds, device="cuda"):
    with torch.no_grad():
        x_0_latent = vae.encode(x_0).latent_dist.sample()

    # Timestep embedding for noise scaling
    def get_timestep_embedding(timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = torch.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    noise_scale = get_timestep_embedding(t, x_0_latent.shape[1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    x_t = x_0_latent + noise * noise_scale

    # Noise prediction
    model_output = model(x_t, t, text_embeds)
    pred_noise = model_output.sample if hasattr(model_output, 'sample') else model_output
    noise_loss = torch.nn.functional.mse_loss(pred_noise, noise, reduction='mean')

    # Flow Matching Loss with UniPC adjustments
    sigma_t = scheduler.sigmas[t].to(device)  # Use scheduler sigmas directly for the timestep
    flow_pred = scheduler.convert_model_output(model_output, sample=x_t, timestep=t)
    v_star = noise / (sigma_t * scheduler.config.flow_shift)  # Adjusted for flow_shift
    flow_loss = torch.nn.functional.mse_loss(flow_pred, v_star, reduction='mean')

    total_loss = noise_loss + 0.1 * flow_loss  # Weight flow loss (adjust as needed)
    return total_loss

# Training Loop
optimizer = torch.optim.AdamW(lora_layers, lr=1e-4)
num_epochs = 2  # Experiment on this one
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