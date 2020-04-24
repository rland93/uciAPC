---
layout: post
title:  "Refactoring"
author: Mike
date:   2020-04-24
categories: update
tags: [visualiation, progress report]
---
## I've been doing some refactoring...

#### Controller:
+ implemented PID controller
+ implemented [a personalization scheme](https://ieeexplore.ieee.org/document/6580276?section=abstract) for setting PID gains

#### Run Script
+ implemented a new batch algorithm that runs in parallel (16x speedup!)
+ changed how data is passed from run script to controller
+ changed structure of output data to make more sense
    + index: time
    + columns: multiIndex, level 0 = category, level 1 = patient name, level 2 = run#

#### Analysis:
+ Charts are now functionally generated, instead of scripted
+ charts now are saved to disk individually rather than in one file

#### Package:
+ Numpy-style docstrings added
+ added requirements.txt

I'm most excited about the new documentation and ~parallell~ ~processing~ that works now. We can run simulations much more accurately now.