#!/bin/bash
while IFS=, read -r start duration output; do
    ffmpeg -i video.mp4 -ss "$start" -t "$duration" -c:v copy -c:a copy "$output"
done < clips.txt