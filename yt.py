from pytube import YouTube

video1 = YouTube("https://www.youtube.com/watch?v=kRHhZ8l9RMQ")
caption = video1.captions['en']
print(caption)
print(caption.generate_srt_captions())