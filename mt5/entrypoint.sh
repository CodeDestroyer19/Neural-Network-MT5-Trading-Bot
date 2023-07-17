#!/bin/bash
# Start the X server
Xvfb :0 -screen 0 1024x768x16 &

# Set the X display
export DISPLAY=:0

# Install necessary dependencies using winetricks
winetricks -q corefonts

# Add a small delay for X server initialization
sleep 2

# Run MetaTrader 5
exec su - trader -c 'wine "/home/trader/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"'

# Clean up X server resources
killall Xvfb
rm -rf /tmp/.X11-unix
