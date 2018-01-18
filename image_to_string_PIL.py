import numpy
import pytesseract
import ImageEnhance
import PIL
import Image

i=6
j=6
flag1='{:0>3}'.format(i)
flag2='{:0>4}'.format(j)
address='/home/charlie/Documents/number/'
image=Image.open(address+'Sample'+flag1+'/img'+flag1+'-0'+flag2+'.png')
# image.show()

# Size Adjusting
newSize=(image.size[0]*5,image.size[1]*3)
image=image.resize(newSize,PIL.Image.ANTIALIAS)

# Property Adjusting
enhancer=ImageEnhance.Color(image)
enhancer=enhancer.enhance(0)
enhancer=ImageEnhance.Brightness(enhancer)
enhancer=enhancer.enhance(2)
enhancer=ImageEnhance.Contrast(enhancer)
enhancer=enhancer.enhance(8)
enhancer=ImageEnhance.Sharpness(enhancer)
enhancer=enhancer.enhance(20)

text=pytesseract.image_to_string(image,lang="eng",config="-psm 7").strip()
print(text)

image.close()
