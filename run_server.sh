#!/bin/bash


export DISPLAY=`/opt/TurboVNC/bin/vncserver -securitytypes tlsnone,x509none,none 2>&1 | perl -0777 -ne '/started on display .*?(:[0-9]*)/ && print $1'`
echo "Started server on display $DISPLAY. To stop it, call stop_server.sh $DISPLAY"
echo
echo "If you did not source this script (ie called via \". run_script.sh\"),
remember to update the DISPLAY variable: export DISPLAY=$DISPLAY"
echo
