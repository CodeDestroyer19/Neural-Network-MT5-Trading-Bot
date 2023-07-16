#!/bin/bash
# Start the X server
Xvfb :0 -screen 0 1024x768x16 &

# Set the X display
export DISPLAY=:0

# Install necessary dependencies using winetricks
winetricks -q corefonts

# Run MetaTrader 5
su - trader -c 'wine "/home/trader/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"'
