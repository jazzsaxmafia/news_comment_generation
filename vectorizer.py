from konlpy.tag import Twitter
import ipdb
import pandas as pd
import os
import re

def cleansing_news(data):
    data['news'] = data['news'].map(lambda news: re.sub(r'\[.*?\]', '', news))
    data['news'] = data['news'].map(lambda news: re.sub(r'\(.*?\)', '', news))
    email_string = data['news'].map(lambda news: re.findall(r'[\w\.-]+@[\w\.-]+', news))

    email_position = map(lambda (news, email): -1 if email == [] else news.index(email[-1]), zip(data['news'].values, email_string.values))

    news_until_email = map(lambda (news, email_pos): news[:email_pos] if email_pos != -1 else news, zip(data['news'].values, email_position))

    data['news'] = news_until_email
    dirty_news = data['news'].map(lambda x: twitter.pos(x))
    clean_news = dirty_news.map(lambda news:
            filter(lambda (word, pos):
                pos != 'Punctuation' and
                pos != 'Foreign' and
                pos != 'URL', news))
    clean_news = clean_news.map(lambda news: ' '.join(map(lambda (word, pos): word, news)))
    data['news'] = clean_news
    return data

def cleansing_comments(comments):
    cleaned_comments = map(lambda comment:
            filter(lambda (word, pos):
                pos != 'Punctuation' and
                pos != 'Foreign' and
                pos != 'URL', twitter.pos(comment)), comments)
    cleaned_comments = map(lambda cleaned_comment: ' '.join(map( lambda (word, pos): word, cleaned_comment)), cleaned_comments)

    return cleaned_comments

twitter = Twitter()

data_path = './data'
data_processed_path = './data_processed'
data_list = os.listdir(data_path)

for data_file in data_list:
#    if os.path.exists(os.path.join(data_processed_path, data_file)):
#        continue
    print data_file
    data_file_path = os.path.join(data_path, data_file)
    data = pd.read_pickle(data_file_path)

    data_news_cleaned = cleansing_news(data)
    data_news_cleaned['comments'] = data_news_cleaned['comments'].map(lambda comments: cleansing_comments(comments))

    data_news_cleaned.to_pickle(os.path.join(data_processed_path, data_file))

