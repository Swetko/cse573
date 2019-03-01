## Installation and basic setup ##
##### Important Note: #####
When signing up for a new cloud account, you get a free $300 credit. This should provide plenty of time (500+ hours of uptime) to work on this project. To be safe, you can stop the instance when you are not using it (select the instance on the “VM Instances” page and press Stop) , as you only get charged while it’s running. If you end up using all of the credit, we have a limited amount of additional credits that we can provide.


#### Set up a cloud account ####
  - Follow the instructions at https://cloud.google.com/free
  - Upgrade to a paid account. There should be an option in the upper right of the console.
  - Request a gpu quota increase
    + Navigation menu > IAM & admin > Quotas
    + Find "GPUs (all regions)" and hit "Edit Quotas". Set the limit to 1
    + This may take a few hours to be approved.

#### Create a compute instance ####
  - Navigation menu > Compute engine > VM instances
  - Select "Create instance" and set the following values:
    + Region: us-west1 (Oregon)
    + Machine type: 1 tesla k80 GPU, 8 cores 30G memory
      * Note: There's a bug when initializing the machine. You are allowed to have up to 8 cores per GPU, but you can only select up to 6 while the "customize" view is open. Returning to the basic view after selecting a GPU will allow you to add 8 cores.
    + Boot disk:
      * Deep Learning Image: PyTorch 1.0.0 and fastai m19 CUDA 10.0
      * Standard persistent disk, 40GB
    + Under Management, security, disks, networking, sole tenancy:
      * Under Metadata, add the key "enable-oslogin" with value "TRUE". This will allow you to add your own ssh key and login from an external terminal.

#### Add some Firewall Rules ####
You'll need to connect to a couple of different ports on the server to view the agent running and the output.

Navigation menu > VPC network > Firewall rules
  - Create Firewall Rule
    + Name: turbovnc (or George, this doesn't actually affect anything)
    + Targets: "All instances in the network"
    + Source IP ranges: 0.0.0.0/0
    + Protocols and ports: tcp: 5900-5910
  - Create Firewall Rule
    + Name tensorboard
    + Targets: "All instances in the network"
    + Source IP ranges: 0.0.0.0/0
    + Protocols and ports: tcp: 6006

#### Connect to the instance ####
You can connect through the browser by clicking on "ssh" for your newly created instance on the VM instances page. This should open a terminal in your browser. Once connected, you can add your ssh key and use the listed "External IP" (referenced below as {external ip}) to connect from a terminal.

The first time you connect, it should ask if you want to install the Nvidia driver, which you do. If you don't see a prompt when you first connect, restart the server (stop/start on the cloud console, or "sudo reboot" in the terminal), wait a minute or so, and reconnect.

#### Login as root ####
Most of the things you will run on the server require root access to work properly. The simplest way to do this is to just login as the root user each time you connect to the server:
```
sudo -i passwd # Set this to whatever you want
su
```

#### Setup the project ####
As root, run the provided setup script. After running this, you will need to reboot the server to allow the configuration changes to take effect. You will have to reconnect, and re-login as root.
```
./setup.sh
reboot
```

#### Running the VNC Server ####
The code uses [Unity](https://unity3d.com) to render the environment. Because it is a headless server, you need to run a vnc server to allow unity to render. This must be done each time you restart the instance. Run the provided script to start the vnc server and then set the DISPLAY. This allows the code to render the Thor environment properly.
```
. run_server.sh # Note the syntax. This sets DISPLAY in the current process as opposed to starting a new one.
```

#### Stopping the VNC Server ####
When you are finished running your code, and want to shut down the instance, you can stop the vnc server by calling
```
./stop_server.sh $DISPLAY # or replace $DISPLAY with its value from above.
```

#### Viewing the environment ####
With everything up to this point, you are able to run the code, but will not be able to see the agent as it interacts with the environment. While this is not strictly necessary, it may be useful, especially in the second part. In order to view the remote graphics, you can use [TurboVNC](https://www.turbovnc.org) viewer.
  - Download and install TurboVNC on your local machine. Note that the download contains both the server and viewer.
  - Assuming you added the firewall rule above, run TurboVNC Viewer and connect to {external ip}:{display number}

#### Running the code ####
Now that everything is set up, you can run the code. The following command will train an agent on a single scene with all objects in fixed locations.
```
vglrun python3 main.py --workers 8 --gpu 0 --scenes 1
```
If you are running the VNC Viewer, you should see a window pop up showing frenetic movements around a kitchen as the agent repeatedly explores the scene trying to find a tomato.

#### Viewing the output ####
The code logs relevant information, including success rates and runtime, while it's running using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard). You can view these logs by running the following code on the server:
```
./run_tensorboard.sh --logdir ./runs
```
Then in a local browser, go to {external ip}:6006 to see the logs. Due to the issue with torch being removed when the server is restarted after installing tensorflow, this script creates a virtualenv, then installs and runs tensorboard from there.
