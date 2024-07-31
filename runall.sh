#!/bin/bash

# Open a new terminal window and run the first command
gnome-terminal -- bash -c "cd IO-Module/; python3 objDetect_server_test.py; exec bash"

# Open a new terminal window and run the second command
gnome-terminal -- bash -c "cd IO-Module/; GST_DEBUG=*:4 python3 flask_client2.py; exec bash"

# Open a new terminal window and run the third command
gnome-terminal -- bash -c "cd IO-Module/; python3 main_server.py; exec bash"

sleep 3

# Open a new terminal window and run the fourth command
gnome-terminal -- bash -c "cd Source-Module/; ffmpeg -re -stream_loop -1 -i 'Truck.ts' -map 0 -c copy -f mpegts 'udp://239.0.0.1:1234?ttl=13'; exec bash"

sleep 2

# Open a new terminal window and run the fifth command
gnome-terminal -- bash -c "cd Source-Module/; ffplay 'udp://127.0.0.1:5555'; exec bash"
