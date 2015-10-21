import urllib
from bs4 import BeautifulSoup
import pandas as pd
import ipdb
import os

from selenium import webdriver

news_prefix = 'http://entertain.naver.com'
date_range = pd.date_range(start='2006-04-01', end='2015-10-11')
driver = webdriver.Chrome()

for date in date_range:

    date_string = date.strftime('%Y-%m-%d')
    if os.path.exists(os.path.join('data', date_string+'.pickle')):
        continue

    print "processing ", date_string
    news_list = []
    comment_list = []
    driver.get('http://entertain.naver.com/ranking#type=hit_total&date='+date_string)
    try:
        href_list = map(lambda x: x.get_attribute('href'), driver.find_elements_by_class_name('tit'))
    except:
        continue

    for href in href_list:

        driver.get(href)
        news_data = BeautifulSoup(urllib.urlopen(href).read())

        news_body = news_data.find('div', {'id':'articeBody'}).text.strip()

        iframe = driver.find_element_by_id('ifrMemo')
        driver.switch_to_frame(iframe)
        comments = map(lambda x: x.text, driver.find_elements_by_class_name('txt'))

        driver.switch_to_default_content()

        news_list.append(news_body)
        comment_list.append(comments)

    dataframe = pd.DataFrame({'news':news_list, 'comments':comment_list})
    dataframe = dataframe[dataframe['comments'].map(lambda x: len(x) > 0)]
    dataframe.to_pickle(os.path.join('data', date_string+'.pickle'))
