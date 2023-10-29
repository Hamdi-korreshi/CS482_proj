import os
from youtube_transcript_api import YouTubeTranscriptApi
import json

if not os.path.exists("Captions"):
    os.mkdir("Captions")

with open("video_id.txt", "r") as file:
    video_ids = [line.strip() for line in file]

for video_id in video_ids:
    try:
        caption_info = YouTubeTranscriptApi.list_transcripts(video_id)
        english_caption = caption_info.find_transcript(['en'])
        if english_caption:
            transcript = english_caption.fetch()
            output_path = os.path.join("Captions", f"{video_id}_transcript.json")
            with open(output_path, "w") as output_file:
                json.dump(transcript, output_file)
            print(f"Transcript for video {video_id} (English) saved as {output_path}.")
        else:
            print(f"No English captions available for video {video_id}.")
    except Exception as e:
        print(f"Failed to retrieve transcript for video {video_id}. Error: {e}")