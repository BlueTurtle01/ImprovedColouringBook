from PIL import Image
import numpy as np

img = Image.open('OutputImages/ash/Threshold Gray.jpg')
img = img.convert("RGBA")

imgnp = np.array(img)
print(imgnp.shape)

white = np.sum(imgnp[:, :, :3], axis=2)
white_mask = np.where(white == 255*3, 1, 0)

alpha = np.where(white_mask, 0, imgnp[:, :, -1])

imgnp[:, :, -1] = alpha

img = Image.fromarray(np.uint8(imgnp))
img.save("OutputImages/ash/Threshold Gray Transparent.png")

def create_alpha():
    original_image = cv2.imread(file_path)
    outline = outline.convert("RGBA")
    print(outline.shape)
