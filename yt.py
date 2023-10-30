from pytubefix import YouTube
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# Replace with the desired Google Drive folder ID
drive_folder_id = '1y2UcjtiYlJX6jKczGW8AOVaOwejPBK-S'

# Grab the current directory
curr_dir = os.getcwd()
directory = '/mnt/c/Users/adema/NJIT/CS482/project/CS482_proj/Videos'

# Read video links from a file
with open('links.txt', 'r') as links:
    for link in links:
        # Create the YouTube object to access everything
        video = YouTube(link)

        # Create the captions file name and the folder name
        captions = video.title + " Captions"
        folder = "Videos"

        # Make the path for captions to be written to and video downloaded into
        folder_path = os.path.join(curr_dir, folder)

        if not os.path.exists(folder):
            os.mkdir(folder)

        try:
            caption = video.captions['en']
            # Modify to make the file inside the folder
            write_captions = os.path.join(folder, captions + ".txt")
            file1 = open(write_captions, "w")
        
            # Make the file in a readable format
            file1.write(caption.generate_srt_captions())
            file1.close()

            # Download video to the new folder
            video.streams.get_by_itag(22).download(output_path=folder_path)
        except KeyError:
            continue
            

for filename in os.listdir(directory):
    file = drive.CreateFile({
        'title': filename,  # Use the local file's name as the title on Google Drive
        'parents': [{'id': drive_folder_id}]  # Set the parent folder
    })    
    file.SetContentFile(os.path.join(directory, filename)) # Set the content to the local file
    file.Upload()