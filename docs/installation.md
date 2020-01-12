---
layout: page
title: "Installation"
permalink: /installation/
---
uciAPC is written in [Python 3.8.1](https://www.python.org/downloads/).

## Windows
Installation of simGlucose was a challenge on my windows machine. Here is what worked:

First, you will need to configure a virtual environment: see https://docs.python.org/3/library/venv.html.

After you have activated the virtual environment, you can try to install simGlucose using pip: ```pip install simGlucose```. This will attempt to automatically resolve dependencies, but if you encounter errors, you will need to configure dependencies manually. Download .whl python packages of ```matplotlib```, ```scipy```, etc (whatever fails to install) from https://www.lfd.uci.edu/~gohlke/pythonlibs/. Check first which python3 you have installed by entering ```python3``` into the terminal. You should see a prompt like this:

 ```>>> Python 3.8.1 (tags/v3.8.1:1b293b6, Dec 18 2019, 22:39:24) [MSC v.1916 32 bit (Intel)] on win32```

 *⚠️ If you see python 2.xx (perhaps 2.7) nothing will work. Go download and configure Python 3.*

You must install the .whl which matches the version of python you have configured. Ie: ```matplotlib‑3.1.2‑cp38‑cp38‑win_amd64.whl``` is matplot lib, version 3.1.2, compiled for python 3.8.xx, 64 bit. Download the preconfigured .whl binary into the project folder.

Double check that things match, and install the wheel using e.g. ```pip install matplotlib‑3.1.2‑cp38‑cp38‑win_amd64.whl```

If that fails, double check that the name matches your python -- if it does not match, download the proper version; if it does match, change the *second* e.g. ```-cp38``` in the filename to ```none```. Pip checks only the filename of the .whl for compatibility, so by changing the name, you can bypass this version check.

You can remove the .whl from the folder after installation.

Repeat this process until all dependencies within the package are resolved.

## MacOS / Linux