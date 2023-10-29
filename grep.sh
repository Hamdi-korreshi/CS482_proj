#!/bin/bash

grep -o 'https://www.youtube.com/watch?v=[^"]*' YouTube.html | sed -e 's/https:\/\/www\.youtube\.com\/watch?v=//' | sort | uniq > video_id.txt

grep -o 'https://www.youtube.com/watch?v=[^"]*' YouTube.html | uniq > links.txt