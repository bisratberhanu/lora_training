import random
import json

# Total duration: 24:36 = 1476 seconds
# Ad segment: 5:22–8:47 = 322–527 seconds
valid_ranges = [(0, 322), (527, 1471)]  # 1471 = 1476–5 to avoid end overflow
num_clips = 15
clip_duration = 5
output_file = "clips.txt"
metadata_file = "metadata.json"

# Generate random start times
start_times = []
while len(start_times) < num_clips:
    # Choose a valid range based on duration weights
    range_lengths = [r[1] - r[0] for r in valid_ranges]
    total_length = sum(range_lengths)
    weights = [length / total_length for length in range_lengths]
    chosen_range = random.choices(valid_ranges, weights=weights, k=1)[0]
    # Pick a start time within the range
    start = random.randint(chosen_range[0], chosen_range[1] - clip_duration)
    # Check for overlap with existing clips
    overlap = any(abs(start - existing) < clip_duration for existing in start_times)
    if not overlap and start not in range(322, 527):
        start_times.append(start)

# Format clips
clips = []
metadata = []
for i, start in enumerate(sorted(start_times), 1):
    # Convert seconds to MM:SS
    minutes = start // 60
    seconds = start % 60
    start_str = f"{minutes:02d}:{seconds:02d}"
    # Generate filename and caption
    filename = f"clip_{i:03d}.mp4"
    # Sample captions (adjust based on episode content)
    caption = f"dragon_ball_z, dragonball_style, anime_style"
    if i % 3 == 0:
        caption += ", goku, super_saiyan, fighting, energy_blast"
    elif i % 3 == 1:
        caption += ", vegeta, armor, punching"
    else:
        caption += ", transformation, glowing_aura"
    clips.append(f"{start_str},{clip_duration},{filename}")
    metadata.append({"file_name": filename, "caption": caption})

# Write clips.txt
with open(output_file, "w") as f:
    f.write("\n".join(clips))

# Write metadata.json
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Generated {output_file} and {metadata_file}")