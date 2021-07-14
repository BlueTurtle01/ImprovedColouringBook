# What does this repo do?
This repo will allow you to upload an image and create a colouring book like outline from the image that can then be given to a child say and allow them to colour in a personalised image - such as a family photo.

# Description of files
 
**ColouringWithMask.py**<br>
First obtains a mask that is fed into the GrabCut algorithm to remove the foreground part of the image which is then
passed to the k-means algorithm and onto the AdaptiveThresholding function.

**ColouringWithRegion.py**<br>
Estimates a bounding box around all the foreground objects using a Faster-RCNN that is fed into the GrabCut algorithm to remove the foreground part of the image which is then
passed to the k-means algorithm and onto the AdaptiveThresholding function.

**Contours.py**<br>
Applies the Canny Edge Detection algorithm, and then an OpenCV find & draw contours method.

**ComicCreator.py**<br>
Extends the outputs of the AdaptiveThresholding function by overlaying the output on top of the clustered output from
the k-means algorithm to create a comic book like effect. A comic book-like caption and text are layered onto the top.

**PaintingUnderlay.py**<br>
Similar to ComicCreator.py but does not include the caption or text. The blend is also semi-transparent. This should serve
as an underlay to guide the person colouring/painting the outline which colour to use. This is similar to paint by numbers
but might be more accessible to younger audiences.

UtilityFunctions.py

