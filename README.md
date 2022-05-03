# bar_tracking
WIP AI barbell tracking, using pytorch and a CNN

main script: <https://github.com/cluffa/bar_tracking/blob/main/src/bar_tracking/track.py>

I created this model and python package to be able to track a barbell and get different metrics. It works using a convolutional neural network with 2 million parameters. This takes a 320x320x3 matrix input and outputs a segmentation of the image (aka mask). Ellipsis are fit to the largest inside and outside weight plates detected in the mask. This is a reliable way find the center, even if the object is partially out of frame. The average of the two sides is used for the metrics. This is a good way to combat some of the distortions due to off-axis movements like rotation. The plates are always a constant 450 mm, so I was able to scale the units from pixels to meters using the dimensions of the ellipsis. The position at every time is then used to create two splines, f1(t)=x and f2(t)=y. The velocity and acceleration are derived from the splines. These also go through Savgov filters remove distortions and noise.

## Current Progress
### Test video  
![test-overlay](./docs/test_out.gif)

### Plot   
![test-plot](./docs/output.png)
