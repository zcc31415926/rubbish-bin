#method: BeautifulSoup
#in python source code:

from bs4 import BeautifulSoup
#BeautifulSoup is a kind of python library for web analysis.
import requests
#requests is a kind of python library for network calling.
#for distant network.
import time
#time is a kind of python library for time control.
#in order not to attract the netservers' interest,
#set a constant sleep time during every two scraping actions.

time.sleep(2)
#second as unit.

#for distant network:
url='http://xxx.html'
#url is the address of the target network.
headers={
    'User-Agent':'User-Agent_name'
    'Cookie':'Cookie_name'
}
#headers stores the login information of a specific user.
#when having forgotten the user account or the password,
#use the structure headers to scrape the information of the account.
#some websites adopt anti-scraping Javascript structures.
#however, not all mobile phones can adjust to these structures.
#therefore, use the User-Agent_name of the mobile phone to solve the problem.
#call the monitor and choose the function 'toggle device toolbar'
#to change the virtual device to mobile phones.
data_name=requests.get(url,headers=headers)
#headers is one of the hidden parameters.
data_name=requests.get(url)
Soup=BeautifulSoup(data_name.text,'lxml')
#data_name is the response from the network.
#text is a method to make the content readable.
#the structures of the webpages are often complex in terms of distant network.
#therefore, find one of the targets, check its parent directory and label,
#and change the CSS_Selector address to 'parent_directory_name > label_name'
#to find all the targets.

#for local webpages:
with open('webpage_name.html','r') as dataset_name:
    Soup=BeautifulSoup(dataset_name,'lxml')
    data_name=Soup.select('data_name_address')
    image_name=Soup.select('image_name_address')
    print(data_name)
#data_name_address is the CSS_Selector-Mode address of the target data.

for data_member_name in data_name:
    print(data_member_name.get_text())
#get_text() is a method to get the text contents in data_member_name.

for image_member_name in image_name:
    print(image_member_name.get('src'))
#get() is a method to get the properties of image_member_name.
#images do not have text contents, but properties (height, width, address, etc.)

print Soup.select(img[width="160"])
#the structure: label_name[statements] can be used
#for selecting the targets satisfying the statements.

print(data_name.stripped_strings)
#stripped_strings is a method to get all the text contents in subaddress members.
