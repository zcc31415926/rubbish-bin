from bs4 import BeautifulSoup
import requests
import time


# in order not to attract the netservers' interest,
# set a constant sleep time during every two scraping actions.
# second as unit.
time.sleep(2)

# for distant webpages
url = 'http://xxx.html'

# headers stores the login information of a specific user.
# when having forgotten the user account or the password,
# use the structure headers to scrape the information of the account.
# some websites adopt anti-scraping Javascript structures.
# however, not all mobile phones can adjust to these structures.
# therefore, use the User-Agent_name of the mobile phone to solve the problem.
# call the monitor and choose the function 'toggle device toolbar'
# to change the virtual device to mobile phones.
headers = {
    'User-Agent':'User-Agent_name',
    'Cookie':'Cookie_name'
}

data_name = requests.get(url, headers=headers)
data_name = requests.get(url)
Soup = BeautifulSoup(data_name.text, 'lxml')

# for local webpages:
with open('webpage_name.html', 'r') as dataset_name:
    Soup = BeautifulSoup(dataset_name, 'lxml')
    data_name = Soup.select('data_name_address')
    image_name = Soup.select('image_name_address')
    print(data_name)

for data_member_name in data_name:
    print(data_member_name.get_text())

for image_member_name in image_name:
    print(image_member_name.get('src'))

print Soup.select(img[width = "160"])
print(data_name.stripped_strings)
