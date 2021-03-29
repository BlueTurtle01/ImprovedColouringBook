from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def white_to_transparency():
    file_name = "sarah2"
    output_path = ("OutputImages/" + str(file_name) + "/")
    img = (str(output_path) + "Threshold Gray.jpg")
    img = Image.open(img)

    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    print(x.shape)

    return Image.fromarray(x)

white_to_transparency()

:param
outline: The
numpy
array
of
outline
pixel
values
:param
clusters: Number
of
clusters
from the k

-means
algorithm
:return: NA
"""
alpha = 1  # This is the amount of the coloured image we include. 0 is none.
beta = 1 - alpha
compressed_image = imread((str(output_path) + str(file_name) + str(clusters) + "Clustered.jpg"))
compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)

# Change the outline to RGBA and then convert to PIL Image
outline_rgba = white_to_transparency()
print(type(outline_rgba))

# Convert the compressed image to RGBA
compressed_rgba = cv2.cvtColor(compressed_image, cv2.COLOR_RGB2RGBA)

#new_img = Image.blend(compressed_PIL, outline_rgba, 0.5)

outline = np.concatenate([outline[..., np.newaxis]] * 3, axis=2)

#output = cv2.addWeighted(outline_rgba, beta, compressed_rgba, alpha, 0)
compressed_rgba.paste(outline_rgba, (0,0), compressed_rgba)


#plot_image(output, "Blended")


def create_alpha():
    from PIL import Image
    outline_rgba = (str(output_path) + "Threshold Gray.jpg")
    outline_rgba = Image.open(outline_rgba)
    outline_rgba = outline_rgba.convert("RGBA")

    outline_rgba_np = np.array(outline_rgba)
    white = np.sum(outline_rgba_np[:, :, :3], axis=2)
    white_mask = np.where(white == 255 * 3, 1, 0)
    alpha = np.where(white_mask, 0, outline_rgba_np[:, :, -1])

    outline_rgba_np[:, :, -1] = alpha

    img = Image.fromarray(np.uint8(outline_rgba_np))
    img.save("OutputImages/ash/Threshold Gray Transparent.png")

    return outline_rgba_np