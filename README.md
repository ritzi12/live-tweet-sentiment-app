# REAL TIME TWEET SENTIMENT ANALYSIS APP
This is a repository of a real time tweet sentiment analysis app built using streamlit platform and using Hugging Face Bertweet Model as core classifier model.

Below is the demonstration of app for #IPL2022 hashtag.

![Recording 2022-03-26 at 19 57 26](https://github.com/ritzi12/live-tweet-sentiment-app/assets/80144294/6a71ffc8-8317-451c-8e17-2e53ca1162e6)
26.gif)

## FEATURES
* This app extracts tweets from twitter apis based on input hashtag ,keyword or twitter user handle .
* It shows sentiment distribution of tweets and highest liked tweet , RT .
* Top twitter handles based on tweets, user mentions etc.
* It also displays word cloud for corresponding sentiment.
* It creates wordcloud for each of the three sentiments based on selected sentiment (positive, neutral, negative)

NOTE: Tweets for free twitter API can fetch only a week old tweets and not beyond it.

### Used plotly for visualisations and wordcloud package for word cloud images .

### Note this repo was deployed on heroku which is no longer available now as Heroku has become paid service after salesforce aquisition of it.

