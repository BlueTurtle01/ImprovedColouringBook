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

<figure>
<img src = "InputImages/race.jpg" width="400">
<img src = "OutputImages/race/Contours.jpg" width = 400>
</figure>

**ComicCreator.py**<br>
Extends the outputs of the AdaptiveThresholding function by overlaying the output on top of the clustered output from
the k-means algorithm to create a comic book like effect. A comic book-like caption and text are layered onto the top.

<img src = "OutputImages/family/Comic Character.png" width = 400>


**PaintingUnderlay.py**<br>
Similar to ComicCreator.py but does not include the caption or text. The blend is also semi-transparent. This should serve
as an underlay to guide the person colouring/painting the outline which colour to use. This is similar to paint by numbers
but might be more accessible to younger audiences.

# Things I learnt from this project
1. I learnt about Canny Edge detection, and how to implement it, from my own research capabilities as I had no knowledge of this algorithm beforehand.
2. I improved my use of Python Classes.
3. I learnt to appreciate more the constraints that having the aim of making the outcome production ready puts onto a project. Theoretically this task is completely possible, but when considering that if this were to be put into production as a busines venture, we would not have unlimited time and resources, and would unlikely know the set of images that the user would want to upload. This does constrain us to only accepting images of a certain type, because we know how the algorithm will respond, and hence from a production point of view is not complete. However, the positive from this is that I have gained more awareness of how teams need to work together. It would be easy for the Data Engineers to say it's completely possible, but the manufacturers will have constraints that are based on the real world situation, and not the theoretical land that the Data Engineer may fall foul of.



