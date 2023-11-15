import os
import cv2

folder = "/workspace/CS482_proj/Videos"
prep = "/workspace/CS482_proj/Frames"
all_count = 0
if not os.path.exists(prep):
    os.mkdir(prep)
    print("MADE")
for video in os.listdir(folder):
    if video.endswith('.mp4'):
        print("Loading and resizing:" , video)
        video_path = os.path.join(folder, video)
        cap = cv2.VideoCapture(video_path)
        count = 0
        # read frame loop
        while True:
            ret, frame = cap.read()
        # Break the loop if no more frames are available
            if not ret:
                break
            # every frame 90th frame because its gonna be way too big othewise and now we are grabbing ever 3 seconds or so
            # int is needed so my frames arent 257.9888887877678
            if count % 180 == 0:
                all_count +=1
                frame_filename = os.path.join(prep, f"frame{all_count}.jpg")
                resized_frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(frame_filename, frame)
            count += 1
        # you the release otherwise it courrpts you're docker and or whole ipynb fil :)
        cap.release()
        print(all_count, "frames saved ")
print("Done")