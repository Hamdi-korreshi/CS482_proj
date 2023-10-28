from pytubefix import YouTube

video1 = YouTube("https://www.youtube.com/watch?v=kRHhZ8l9RMQ")
title = video1.title + " Captions"
file1 = open(title, "w")
caption = video1.captions['en']
file1.write(caption.generate_srt_captions())
video1.streams.get_by_itag(22).download()