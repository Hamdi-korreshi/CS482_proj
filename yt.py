from pytubefix import YouTube
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# grab the current directory 
curr_dir = os.getcwd()
# grab the links and iter through them one by one
with open('links.txt', 'r') as links:
    for link in links:
        # create the youtube object to access everything
        video = YouTube(link)
        # create the cpations file name and the folder file name
        captions = video.title + " Captions"
        folder = video.title + " Folder"
        # make the path for captions to be written to and video downloaded into 
        folder_path = os.path.join(curr_dir, folder)
        if not os.path.exists(folder):
            os.mkdir(folder)
        # modify to make the file inside the folder
        write_captions = os.path.join(folder, captions+".txt")
        file1 = open(write_captions, "w")
        caption = video.captions['en']
        # make the file in a readable format
        file1.write(caption.generate_srt_captions())
        # download video to new folder
        video.streams.get_by_itag(22).download(output_path=folder_path)