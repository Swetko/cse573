#!/bin/bash

# Install xorg
echo "y" | apt-get install xorg openbox

# Set up the config file
BUSID=`nvidia-xconfig --query-gpu-info | perl -0777 -ne '/PCI BusID : (.*)/ && print $1;'`
nvidia-xconfig --allow-empty-initial-configuration --use-display-device=None --virtual=1920x1200 --busid $BUSID

# Install and configure VirtualGL
wget -O virtualgl.deb https://sourceforge.net/projects/virtualgl/files/2.6.1/virtualgl_2.6.1_amd64.deb
dpkg -i virtualgl.deb
echo "y" | apt-get install lightdm
/etc/init.d/lightdm stop
echo "1
y
y
y
x" | /opt/VirtualGL/bin/vglserver_config

# Install TurboVNC
wget -O turbovnc.deb https://sourceforge.net/projects/turbovnc/files/2.2.1/turbovnc_2.2.1_amd64.deb
dpkg -i turbovnc.deb

# Install Thor
pip3 install ai2thor setproctitle tensorboardX

# Grab the executable for Thor
wget https://courses.cs.washington.edu/courses/cse573/19wi/project/builds.tar.gz
tar -xzf builds.tar.gz
mv builds datasets
rm builds.tar.gz

# Reboot to allow new configurations to take effect
# reboot

