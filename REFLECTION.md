#**Reflection of Project 1 - Finding Lane Lines on the Road** 
<img src="laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Reflection describes the current pipeline, identifies its potential shortcomings and suggests possible improvements. 

The project provides with a clear structure to process images/videos, using canny detecting edges and using hough lines to plot the lane lines

I created two masks ( left and right) to detect lane lines to avoid lack of resolusion of Hough transfer at the far end of the rode 

possible improvements could be:

1. Adding color detection as a filter
2. Using linear regression and confine the masked area in real time to apply hough transfer

