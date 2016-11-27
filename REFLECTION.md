#**Reflection of Project 1 - Finding Lane Lines on the Road** 
<img src="laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Reflection describes the current pipeline, identifies its potential shortcomings and suggests possible improvements. 

The project provides with a clear structure to process images/videos, using canny detecting edges and using hough lines to plot the lane lines

I created two masks ( left and right) to detect lane lines to avoid lack of resolusion of Hough transfer at the far end of the rode 

potential shortcomings: 

1. In occations when only applying canny could not differencitate to much such as there is some damage or while/yellow marks on the road

2. The program is confited to a predifined parameter for canny, color selection or hough transfer, however, the real road image is unforseen, so a smarter program should be able to detect the color of the lane within a predefined mask and then apply certain parameters based on the image it processed and update those parameters within certain time.

3. when there is a curved line, there should be more masks covering different areas of the lane lines which could better capture the curved lane lines

possible improvements could be:

1. Adding color detection as a filter, hsv would be a good function to try
2. Using linear regression and confine the masked area in real time to apply hough transfer
3. Adding angle filters to the lines selected using hough transfer


