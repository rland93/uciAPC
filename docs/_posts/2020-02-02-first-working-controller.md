---
layout: post
title:  "First working controller, and visualizations"
author: Mike
date:   2020-02-01
categories: update
---

## Controller
We have a controller! For now, it is a very naive PID controller, with no derivative and no integral component. The controller gain is hard-coded for patient 12. It also has a basal suspension feature, which will automatically shut off basal if the blood glucose level drops below a certain value. This is already a common feature on commercial pump/CGM systems.

[![Controller]({{ "ProportionalOnlyController.png" | absolute_url  }})]({{ "ProportionalOnlyController.png" | absolute_url  }})

Our controller has noticeable post-prandial[1] dips -- this one lasts for nearly 4 hours and takes suspension of the pump for that long to fix it. Going forward, we want to minimize those dips as much as possible.

## Plotting functionality

I also added a plotting function, so that plots are automatically generated on simulation. Evaluating the data from the simulation, especially from multiple simulations with CGM noise, and making beautiful plots automatically will be another area of development for the project.

[1] after-meal