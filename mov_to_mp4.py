from moviepy import VideoFileClip

# Charger la vidéo .mov
clip = VideoFileClip("pics/felix_vid.mov")

# Exporter en .mp4
clip.write_videofile("pics/felix.mp4", codec="libx264")
