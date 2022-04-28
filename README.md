# bar_tracking
WIP AI barbell tracking, using pytorch and a CNN

## Current Progress
### Test video with tracked plates, centers, and average  
The highlights on the plates are segments output by the model. The center of the plates was found using expected value of the marginal distributions of red and green pixels on each axis. The red dot is the average of the two sides.  
![test-overlay](./docs/test_out.gif)

### X-Y Position Plot
This shows that the tracking can be very accurate. The bar path is calculated from the center red dot on the gif above. The units have been scaled automatically based on the size of the plates from the video.   
![test-plot](./docs/output.png)