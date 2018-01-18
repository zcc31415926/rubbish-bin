from PIL import Image
image=Image.open('/home/charlie/Pictures/Wallpapers/vim.png')

# display the input image
# image.show()
# image.convert('L').show()

# convert the input image to the JPG format
# the saving process will fail without the color_format conversion
# image2=image.convert('RGB').save("vim.jpg")

# create a thumbnail
# image.thumbnail((128,128))
# image.show()

# cut a region out of the image
# and combine with paste
# box=(100,100,400,400)
# region=image.crop(box)
# region.show()
# region=region.transpose(Image.ROTATE_180)
# image.paste(region,box)
# image.show()

# adjust size and angle
# image.resize((1024,1024)).show()
# image.rotate(45).show()

image.close()
