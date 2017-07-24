import numpy
import pytesseract

image=f.open('/home/charlie/Pictures/image.jpg')
text=pytesseract.image_to_text(image)
print(text)
