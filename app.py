"""
Created on 19-03-2022

@author : Ritika Gupta
"""

import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import utility




def main():
    """ Function to run Streamlit app """
    # ---------------------------HEADER SECTION ----------------------------------------

    st.set_page_config(layout="wide")

    if "first_time_load" not in st.session_state:
        # set the initial default value of the flag
        st.session_state.first_time_load = True
    html_temp = """
	    <div style="background-image: linear-gradient(to right, mediumturquoise, gold);"><p style="color:white;font-family:Poppins, serif;font-size:40px;padding:10px;text-shadow: 2px 2px #000000">Real-time Tweet Sentiment Analysis</p></div>
	    """
    col01,col02 = st.columns([2,15])
    with col01:
        st.image("twitter_logo.png", width=100)
    with col02:
        st.markdown(html_temp, unsafe_allow_html=True)
    with st.expander(label="While the model loads ! You can go through. How to use this dashboard?"):
        st.markdown('''Using this app one can easily fetch and analyse sentiments of tweets 
                In the *sidebar panel* one can search tweets using twitter handle of any account or hashtag/keyword.''')
        st.write(''' **Pie Chart** : Shows percentage distribution of sentiments in the fetched tweets  <br>
                 **Histogram**: Shows the confidence distribution of sentiments. The higher the confidence 0.9~1 gives more confident prediction.  <br>
                 **Timeline Distribution of Tweets**: It shows distribution of tweets based on tweet posted time.  <br>
                 **Violin Chart**: It shows the concentration of likes,retweets and followers. One can also check tweet which are outliers having highest number of likes/RT/Followers <br>
                 **Top N Counts** : One can check which twitter handles/Hashtags/User Mentions have have tweeted highest number of tweets from among fetched tweets  <br>
                 **WordCloud** : Finally we can generate wordclouds for corresponding sentiments and also see random tweets for that sentiment''' , unsafe_allow_html=True)

    with st.spinner("Loading the Model...."):
        model = utility.load_transformer_model()
    # ---------------------SIDEBAR SECTION --------------------------------------------
    st.sidebar.title("Set Arguments")
    tweetFrom = st.sidebar.radio("Fetch Tweets From",options=['Twitter Handle','Hashtag/Keyword'],index=1)
    Topic = st.sidebar.text_input(label="Enter keyword/hashtag/twitter handle to get Sentiment:",
                                  placeholder="Ex : #Twitter , Apple, elonmusk")
    Count = st.sidebar.text_input(label="No. of Tweets to fetch:",
                                  placeholder="by default it's max value set to 50000")
    st.sidebar.text("Filter Tweets based on Dates :")
    since = st.sidebar.text_input(label="From Date (YYYY-MM-DD) ",
                                  placeholder="Ex: 2018-11-12")
    until = st.sidebar.text_input(label="To Date (YYYY-MM-DD) ",
                                  placeholder="Ex: 2021-11-12")
    # ---------------------BODY SECTION - VISUALIZATIONS --------------------------------------------
    def load_all_viz(data):
        col1, col2 = st.columns(2)
        col1.plotly_chart(utility.viz_pie(data),use_container_width = True)
        col2.plotly_chart(utility.viz_hist_confi(data), use_container_width=False)

        col8, col9 = st.columns([40, 1])
        with col8:
            st.write('Timeline of Tweets Sentiments')
            st.plotly_chart(utility.viz_time_series(data), use_container_width=True)

        col3,col4,col5 = st.columns([5,1,3])
        viol_var=col3.selectbox('Show Distribution of ',['Likes', 'RT', 'Followers_Count'])
        col3 = col3.plotly_chart(utility.viz_violin(data,viol_var), use_container_width=True)

        with col4:
            st.empty()  #empty container for layout purpose

        with col5:
            top_counts_val = st.selectbox('Top N Twitter Handles/ Usermentions/ Hashtags (Based on no.of tweets) ',['TwitterHandle','Hashtags','User_mentions','RT','Followers_Count'])
            top_n = st.slider('Pick "N" value for "Top N"', 1, 30)
            if top_counts_val in ['TwitterHandle','Hashtags','User_mentions']:
               show_hist =  st.button("Show Distribution")
            st.table(utility.top_counts(data,top_counts_val, top_n))
        # if button clicked
        if show_hist:
            col6, col7 = st.columns([30, 1])
            with col6:
                st.plotly_chart(utility.viz_show_distribution(top_counts_val, top_n),use_container_width=True)
        col10,col11 = st.columns([1, 2])
        with col10:
            wordcloud_in = st.radio("Display Wordcloud for which sentiment?",options=['Positive','Neutral','Negative'],index=0)
        with col11:
            st.text(f"Random tweets of {wordcloud_in} sentiment")
            senti_map = {'Positive': "POS", 'Neutral': 'NEU', 'Negative': "NEG"}
            selected_senti = senti_map[wordcloud_in]
            st.markdown(data.query('Sentiment_Transformer == @selected_senti')[["Tweet"]].sample(n=1).iat[0,0])
            st.markdown(data.query('Sentiment_Transformer == @selected_senti')[["Tweet"]].sample(n=1).iat[0,0])
            st.markdown(data.query('Sentiment_Transformer == @selected_senti')[["Tweet"]].sample(n=1).iat[0, 0])
        if st.button("Show Wordcloud"):
            utility.get_wordcloud(data,wordcloud_in)

    # ---------------------SIDEBAR SECTION --------------------------------------------
    if st.sidebar.button('Submit'):
        my_bar = st.progress(0.0)  # To track progress of Extracted tweets
        date_map = dict()
        if since:
            date_map['since'] = since
        if until:
            date_map['until'] = until
        if tweetFrom == 'Twitter Handle':
            input = 'from:'+Topic
            data = utility.get_tweets(my_bar, input, Count,**date_map)
        else:
            data = utility.get_tweets(my_bar, Topic, Count,**date_map)
        with st.spinner("Fetching Tweets Sentiments ...."):
            data = utility.get_sentiments(data,model) # Classify Tweets Sentiments
        st.success("Done!")
        # initialize session state variable
        if "data" not in st.session_state:
            st.session_state.data = data
        elif "data" in st.session_state:
            st.session_state.data = data

        st.session_state.first_time_load = False
        load_all_viz(data)
    elif st.session_state.first_time_load:
          pass
    else:
          load_all_viz(st.session_state.data)


if __name__ == '__main__':
    main()


