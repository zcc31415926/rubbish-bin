import numpy
import pytesseract

sum=0
for i in range(1,1000):
    image=f.open('/home/charlie/Pictures/image.jpg')
    text=pytesseract.image_to_text(image)
    if text==name:
        sum++
print(sum/1000)
