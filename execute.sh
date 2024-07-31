#!/bin/bash

# Close all terminal windows running the specific commands

# Close any gnome-terminal processes (adjust if you use a different terminal emulator)
pkill -f "cd IO-Module/; python3 objDetect_server_test.py; exec bash"
pkill -f "cd IO-Module/; python3 flask_client2.py; exec bash"
pkill -f "cd IO-Module/; python3 main_server.py; exec bash"
pkill -f "cd Source-Module/; ffmpeg -re -stream_loop -1 -i Truck.ts -map 0 -c copy -f mpegts 'udp://239.0.0.1:1234?ttl=13'; exec bash"
pkill -f "cd Source-Module/; ffplay 'udp://127.0.0.1:5555'; exec bash"
