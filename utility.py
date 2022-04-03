import pandas as pd
import plotly.express as px
import config
import plotly.graph_objects as go
import tweepy
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator   #https://www.lfd.uci.edu/~gohlke/pythonlibs/#wordcloud
from PIL import Image
import numpy as np
"""
This file contains all modeling and visualization functions
"""
global df
colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen'] #color scheme map

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_transformer_model():
    from transformers import pipeline
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    return classifier


def get_tweets(my_bar,Topic, Count: int, **kwargs):
    consumer_key = config.API_KEY  # API KEY
    consumer_secret = config.API_KEY_SECRET # API KEY SECRET
    access_token = config.ACCESS_TOKEN
    access_token_secret = config.ACCESS_TOKES_SECRET

    # Now we use above credentials to authenticate the API OAuth
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    i = 0
    count_int = int(Count)
    # A data frame to store tweets in proper format one can add/remove columns as desired
    global df
    df = pd.DataFrame(
        columns=["Date", "TwitterHandle", "IsVerified", "Tweet", "Likes", "RT", 'User_location', "Followers_Count",
                 "User_mentions", "Hashtags"])
    for tweet in tweepy.Cursor(api.search, q=Topic, count=Count, lang="en", exclude='retweets',
                               include_entities=True, **kwargs).items():
        # time.sleep(0.1)

        my_bar.progress((i+1)/count_int)
        df.loc[i, "Date"] = tweet.created_at
        df.loc[i, "TwitterHandle"] = tweet.user.screen_name
        df.loc[i, "IsVerified"] = tweet.user.verified
        df.loc[i, "Tweet"] = tweet.text
        df.loc[i, "Likes"] = tweet.favorite_count
        df.loc[i, "RT"] = tweet.retweet_count
        df.loc[i, "User_location"] = tweet.user.location
        df.loc[i, "Followers_Count"] = tweet.user.followers_count
        # df.loc[i,"IsSensitive"] = tweet.user.
        hashtag = []
        user_mention = []
        for item in tweet.entities['hashtags']:
            hashtag.append(item['text'])
        for item in tweet.entities['user_mentions']:
            user_mention.append(item['screen_name'])
        df.loc[i, "User_mentions"] = user_mention
        df.loc[i, "Hashtags"] = hashtag
        i = i + 1
        if i >= int(Count):
            #df.to_pickle("TweetPickle")  # Storing df in pickled form to avoid conversion of lists in to strings
            #df.to_csv('TweetDataset.csv', index=False)  ## Save as Excel
            break
        else:
            pass
    return df


def get_sentiments(data, classifier):
    scores = data.Tweet.apply(lambda x: classifier(x))
    #Store sentiments and confidence in columns
    data['Sentiment_Transformer'] = [score[0]['label'] for score in scores]
    data['Sentiment_Confidence'] = [score[0]['score'] for score in scores]
    return data

@st.cache(show_spinner=False)
def viz_pie(data):
    countsum=data['Sentiment_Transformer'].value_counts()
    countsum.name = "Sentiment Count"
    fig = px.pie(countsum, values="Sentiment Count", names=countsum.index, title="Distribution of Sentiment across Tweets")
    fig.update_traces(textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#ffffff', width=2)))
    return fig

@st.cache(show_spinner=False)
def viz_hist_confi(data):
    #df = pd.read_pickle('sentimentData')
    #colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    fig = px.histogram(data, x="Sentiment_Confidence", color="Sentiment_Transformer",
                       title="Distribution of Sentiment across Tweets", color_discrete_sequence=colors,
                       labels={'Sentiment_Confidence':'Sentiment Confidence'}) # can specify one label per df column
    fig.update_traces(textfont_size=20,marker=dict(line=dict(color='#fafafa', width=1)))
    return fig

@st.cache(show_spinner=False)
def viz_violin(data, viol_var):
    fig = px.violin(data, y=viol_var, box=True, points="outliers",color_discrete_sequence=['darkorange','mediumturquoise'],
                    hover_data=[viol_var,"Tweet","Sentiment_Transformer"])# 'TwitterHandle', 'Sentiment_Transformer', 'Tweet'])
    #fig.add_trace(go.Violin(y=df['RT'],
    #                       legendgroup='Retweets', scalegroup='Retweets', name='Retweets', points="outliers",
    #                        line_color="darkorange"))  # hover=['Likes','TwitterHandle','Sentiment_Transformer','Tweet']))
    # fig.update_layout(hover_data = ['Likes','TwitterHandle','Sentiment_Transformer','Tweet'])
    return fig

@st.cache(show_spinner=False)
def top_counts(data, colname = 'TwitterHandle',top_N='5'):
    if colname == 'Hashtags':
       global hastags_df
       hastags_df = data[[colname, 'Sentiment_Transformer']].explode('Hashtags')
       results = hastags_df[colname].value_counts().nlargest(top_N)
    elif colname == 'User_mentions':
       global userMentions_df
       userMentions_df = df[[colname, 'Sentiment_Transformer']].explode(colname)
       results = userMentions_df[colname].value_counts().nlargest(top_N)
    else:
        results = data[colname].value_counts().nlargest(top_N)
    return results

@st.cache(show_spinner=False)
def viz_show_distribution(colname, top_N):
    if colname == 'Hashtags':
        final = hastags_df[hastags_df[colname].isin(hastags_df.Hashtags.value_counts().nlargest(top_N).index.to_list())]
        fig = px.histogram(final, x=colname, color='Sentiment_Transformer', barmode="overlay", color_discrete_sequence=colors)
        return fig
    elif colname == 'User_mentions':
        final = userMentions_df[userMentions_df[colname].isin(userMentions_df.User_mentions.value_counts().nlargest(top_N).index.to_list())]
        fig = px.histogram(final, x=colname, color='Sentiment_Transformer', barmode="overlay", color_discrete_sequence=colors)
        return fig
    elif colname == 'TwitterHandle':
        final = df[
            df[colname].isin(df.TwitterHandle.value_counts().nlargest(top_N).index.to_list())]
        fig = px.histogram(final, x=colname, color='Sentiment_Transformer', barmode="overlay",
                           color_discrete_sequence=colors).update_xaxes(categoryorder='total descending')
        return fig


def viz_time_series(data):
    fig = px.histogram(data[['Date', 'Sentiment_Transformer']], x="Date", color='Sentiment_Transformer',
                       barmode="overlay", color_discrete_sequence=colors)
    fig.update_traces(textfont_size=20, marker=dict(line=dict(color='#fafafa', width=1)))
    return fig

def get_wordcloud(data,sentiment):
    map = {'Positive': "POS", 'Neutral':'NEU', 'Negative':"NEG"}
    sub = data[data['Sentiment_Transformer'] == map[sentiment]]
    sub['clean_tweet'] = sub['Tweet'].apply(lambda x: clean_tweet(x))
    words = ' '.join(sub['clean_tweet'])

    pic_mask = np.array(Image.open('colorwheel.png'))
    wc = WordCloud(stopwords=STOPWORDS,
                          background_color='black', height=450, width=600,
                          mask=pic_mask,max_words=2000)
    wc.generate(words)
    # create coloring from image
    image_coloring = ImageColorGenerator(pic_mask)
    fig = plt.figure(figsize=(7,9))
    plt.imshow(wc.recolor(color_func=image_coloring), interpolation="bilinear")

    #image = plt.imshow(wordcloud)
    plt.axis('off')
    #plt.xticks([])
    #plt.yticks([])
    st.pyplot(fig)

def clean_tweet(tweet):
    import re
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z\s])|(\w+:\/\/\S+)|([RT])', ' ', str(tweet).lower()).split())

