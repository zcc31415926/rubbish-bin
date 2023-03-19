from PIL import Image


img_path = './Pictures/ubuntu.jpeg'
image = Image.open(img_path)

# display the input image
image.show()
image.convert('RGB').show()
image.convert('L').show()

# convert the input image to the JPG format
# the saving process will fail without the color_format conversion
# image2 = image.convert('RGB').save('ubuntu2.jpg')

# create a thumbnail
image.thumbnail((128, 128))
image.show()

# cut a region out of the image
# and combine with paste
box = (100, 100, 400, 400)
region = image.crop(box)
region.show()
region = region.transpose(Image.ROTATE_180)
image.paste(region, box)
image.show()

# adjust size and angle
image.resize((512, 256)).show()
image.rotate(45).show()

image.close()

