import cv2
import os
import show_bbox_api

# Path to the directory containing images
image_folder = '/home/apg/workspace/event-based-skyview/output/images'
label_folder = '/home/apg/workspace/event-based-skyview/output/rgb_labels'

# Path and name of the output video file
video_filename = 'output_video_rgb.avi'
frame_rate = 10

# Get list of image files
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]

# Sort images by filename (assumes filenames are sequentially numbered)
images.sort()

# Check if images list is not empty
if not images:
    raise ValueError("No images found in the specified directory.")

# Read the first image to get its dimensions
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec for .avi files
video = cv2.VideoWriter(video_filename, fourcc, frame_rate, (width, height))

# Write images to video
for image in images:
    img_path = os.path.join(image_folder, image)
    label_name = os.path.splitext(image)[0]+".txt"
    label_path = os.path.join(label_folder, label_name)

    # show_bbox.draw_frame(img_path label_path)
    # frame = cv2.imread(img_path)
    video.write(show_bbox_api.draw_frame(img_path, label_path))

# Release the video writer object
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {video_filename}")
