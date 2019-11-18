from urllib.request import urlopen
from io import StringIO
from bs4 import BeautifulSoup
import csv
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager,process_pdf
from pdfminer.layout import LAParams


# print pure text files
textPage = urlopen("http://www.pythonscraping.com/pages/warandpeace/chapter1-ru.txt")
print(str(textPage.read(), 'utf-8'))

# use BeautifulSoup and Python3 to change the code format into 'utf-8'
html = urlopen("http://en.wikipedia.org/wiki/Python_(programming_language)")
bsObj = BeautifulSoup(html)
content = bsObj.find("div", {"id":"mw-content-text"}).get_text()
content = bytes(content, "utf-8")
content = content.decode("utf-8")

# print csv files
data = urlopen("http://pythonscraping.com/files/MontyPythonAlbums.csv").read().decode('ascii', 'ignore')
dataFile = StringIO(data)
csvReader = csv.reader(dataFile)
for row in csvReader:
    print(row)
# use iterator
for row in csvReader:
    print("The album \""+row[0]+"\" was released in "+str(row[1]))
# use DictReader
dictReader = cv.DictReader(dataFile)
print(dictReader.fieldnames)
for row in dictReader:
    print(row)

# print pdfs
def readPDF(pdfFile):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    process_pdf(rsrcmgr, device, pdfFile)
    device.close()

    content = retstr.getvalue()
    retstr.close()
    return content

pdfFile = urlopen("http://pythonscraping.com/pages/warandpeace/chapter1.pdf")
outputString = readPDF(pdfFile)
print(outputString)
pdfFile.close()
