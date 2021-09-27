import pandas as pd
import string
import re
from nltk.corpus import stopwords
import numpy as np
import nltk
from selenium import webdriver
import time
import yfinance as yf
from transformers import pipeline
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

# Streamlit Deployment Code

companies=['None','01:INFOSYS', '02:TCS', '03:RELIANCE INDUSTRIES', '04:ICICI BANK' , '05:HDFC Bank', '06:HDFC', '07:BHARTI AIRTEL' ,  '08:Wipro' ,
           '09:SBI' ,'10:Yes Bank' , '11:Tata Motors','12:Vodafone Idea','13:PNB', '14:MARUTI SUZUKI','15:ONGC', '16:TATA STEEL', '17:NTPC']

st.set_page_config(page_title="NLP applications in Finance", page_icon=None, layout='wide', initial_sidebar_state='expanded')
#image = Image.open('/content/IMG2.png')
#st.image(image, use_column_width=True)


st.sidebar.subheader("About us")
#st.sidebar.title("About us")
st.sidebar.write('''
Innodatatics is actively transformed by its highly extreme entities with various decades of realm 
expertise among its representatives. Our R&D team advances to conceive by collaborating with its alma maters 
in finding solution industry-complicated issues.''')

st.sidebar.subheader("Project")
st.sidebar.write("Measuring Impact of Financial News on Stock Market")
select = st.sidebar.selectbox( "Choose a company?",(companies))
st.sidebar.subheader("Contact us")
st.sidebar.write("Innodatatics Inc")
st.sidebar.write("[Website](https://innodatatics.ai)")
st.sidebar.write("© Copyrights 2021 Innodatatics")

if select=='None':
  st.write("You haven't selected any company yet")
  
else:
  chrome_options = webdriver.ChromeOptions()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')
  driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)

  st.write(f"You have selected: {select[3:]}")
  
  x=int(select[:2])
  company={ 1: ['INFOSYS','INFY.NS'], 2:['TCS','TCS.NS'] , 3: ['RELIANCE INDUSTRIES','RELIANCE.NS'],
  4: ['ICICI BANK','ICICIBANK.NS'] , 5: ['HDFC BANK','HDFCBANK.NS'], 6: ['HDFC','HDFC.NS'],
  7:['BHARTI AIRTEL','BHARTIARTL.NS'],  8: ['Wipro','WIPRO.NS'], 9: ['SBI','SBIN.NS'],
  10: ['Yes Bank','YESBANK.NS'],11:['Tata Motors','TATAMOTORS.NS'],12:['Vodafone Idea','IDEA.NS'],
  13:['PNB','PNB.NS'], 14: ['MARUTI SUZUKI','MARUTI.NS'],15:['ONGC','ONGC.NS'],
  16: ['TATA STEEL','TATASTEEL.NS'], 17: ['NTPC','NTPC.NS'] }

  company_name=company[x][0]
  company_stock_code=company[x][1]

  url='https://www.livemint.com/'

  driver.get(url)
  driver.find_element_by_class_name("iconSprite").click()
  driver.find_element_by_name("searchParameter").send_keys(company_name)
  driver.find_element_by_name("btnSearch").click()

  for i in range(0,40,1):
      driver.execute_script("window.scrollBy(0,750)","")
      time.sleep(1)

  # Extracting Headlines
  list1=driver.find_elements_by_class_name("headline")
  list1=[list1[i].text for i in range(len(list1))]

  # Extarcting dates
  dates=driver.find_elements_by_css_selector(".date >span")

  list2=[]
  count=0
  for date in dates:
      id1=date.get_attribute("id")
      search=driver.find_element_by_id(id1)
      x=search.get_attribute("data-updatedtime")
      list2.append(x)
      count+=1
      # As extra links get attached for dates attributing
      # so making an codition to limit those dates to only number of headlines extracted
      if count==(len(list1)):
          break

  list2=[str(list2[i]) for i in range(len(list2))]
  list3=[list2[i][:10] for i in range(len(list2))]


  news_data=pd.DataFrame(columns=["Date","Title"])
  news_data["Date"]=list3
  news_data["Title"]=list1
    
  #st.write(news_data.head())
  #st.write(news_data.tail())  

  #### Yahoo Finance Data
  ## company stock price data
  
  fin_data = yf.download(company_stock_code, start=list3[-1], end=list3[0])
  fin_data =fin_data[["Close"]]
      
  fin_data=fin_data.reset_index()
  fin_data['Date']=[str(fin_data['Date'][i].date()) for  i in range(len(fin_data))]

  #st.write(fin_data.head())
  #st.write(fin_data.tail())

  ##
  data=news_data
  data1=fin_data

  data=data[::-1].reset_index()
  data.drop(columns=['index'],axis=True,inplace=True)
  ##

  # Cleaning of Data
  def clean_data(data):
    
    list1=[]
    list1=data['Title']
    import re
    import nltk
    nltk.download('stopwords')
    
    for i in range(len(list1)):               
        tokens = list1[i].split()
        # prepare regex for char filtering
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        # remove punctuation from each word
        tokens = [re_punc.sub('', w) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        list1[i] = ' '.join(tokens)
    data['Title']=list1
    return data

    data=clean_data(data)
      


  # MODEL:

  def model_sentiment(data):
      classifier = pipeline('sentiment-analysis', model="ProsusAI/finbert")
      list3=[]
      for i in range(len(data['Title'])):
          a=(classifier(data['Title'][i]))
          a=a[0]
          a=list(a.values())[0]
          label_f={'positive':1,'negative':2,'neutral':0}
          list3.append(label_f[a])
      data['1day_sentiment']=list3
      del list3
      return (data)
    

  #As data has many news articles published in a day we need to aggregate all the news article on a particular day.
  data=model_sentiment(data)

  aggregation_functions = {'Title': 'first', '1day_sentiment': 'max'}
  data = data.groupby(data['Date']).aggregate(aggregation_functions)
  data.reset_index(inplace=True)

  data['1day_sentiment']=data['1day_sentiment'].astype('float')

  # Calculating Exponential Weighted average for 3,7,15 days sentiments
  data['3day_sentiment'] = round(data['1day_sentiment'].ewm(span=3).mean())
  data['7day_sentiment'] = round(data['1day_sentiment'].ewm(span=7).mean())
  data['15day_sentiment'] = round(data['1day_sentiment'].ewm(span=15).mean())

  data2=data.drop(columns=['Title'],axis=1)# Dropping Title as for further analysis, it's not required.

  # Labelling the data for better understanding
  
  label={0: 'Neutral',1:'Positive',2:'Negative'}
  data2['1day_sentiment']=[label[data2['1day_sentiment'][i]]for i in range(len(data2))]
  data2['3day_sentiment']=[label[data2['3day_sentiment'][i]]for i in range(len(data2))]
  data2['7day_sentiment']=[label[data2['7day_sentiment'][i]]for i in range(len(data2))]
  data2['15day_sentiment']=[label[data2['15day_sentiment'][i]]for i in range(len(data2))]
  
  # Calculating Percentage change

  data1['perc_change1'] =data1['Close'].pct_change(periods=1)*100
  list1=list(data1.perc_change1)
  list1.append(np.nan)
  del list1[0] # for 1 forward window del 1st value
  data1['rolling_perc']=list1
  data1.drop(columns=['perc_change1'],axis=1,inplace=True)

  # Converting dates from string to date format for merging two columns:

  data2['Date']=pd.to_datetime(data2['Date'],format='%Y-%m-%d')
  data1['Date']=pd.to_datetime(data1['Date'],format='%Y-%m-%d')

  # Merging the data frame per date column in a sequential order
  merge = pd.merge_asof(data2, data1, on='Date')

  # Last rows of the dataframe will not have any percentage values as there is no new data is available. So saving those indexes in list to avoid error in 
  # future analysis.

  null_perc_change= list(np.where(merge['rolling_perc'].isnull())[0])


  # Business Problem Analysis
  # A. Sentiment Meter
  # B. Price Movement analysis based on Sentiment meter
  # To Study how many investment oppurtunities were profitable, loss making
  data4=pd.DataFrame(columns=["Description","Total Opportunities","Profit(%)","Loss(%)","No Profit No Loss(%)"])
  desc=[]
  total1=[]
  profit1=[]
  loss1=[]
  npnl1=[]
  #These are done to compute how sentiment rolling days is behaving with actual price
  col=[1,2,3,4]
  for j in col:
    #print(j)
    profit=0
    loss=0
    npnl=0
    for i in range(len(merge)):
      if i in null_perc_change:
        continue
      #print(i)
      if merge.iloc[i,j]=='Negative' and  merge.iloc[i,6]<0:
        profit+=1
      elif merge.iloc[i,j]=='Negative' and  merge.iloc[i,6]>0:
        loss+=1   
      elif merge.iloc[i,j]=='Positive' and  merge.iloc[i,6]>0:
        profit+=1
      elif merge.iloc[i,j]=='Positive' and  merge.iloc[i,6]<0:
        loss+=1
      if merge.iloc[i,j]=='Neutral':
        npnl+=1
    total= profit + loss +npnl
    desc.append(j)
    total1.append(total)
    profit1.append(profit/total*100)
    loss1.append(loss/total*100)
    npnl1.append(npnl/total*100)

  #Summarizing the dataframes
  data4["Description"]=["Rolling 1 day-News Day Event","Rolling 3 day-News Day Event","Rolling 7 day-News Day Event ","Rolling 15 day-News Day Event"]
  data4["Total Opportunities"]=total
  data4['Profit(%)']=profit1 
  data4['Loss(%)']=loss1
  data4['No Profit No Loss(%)']=npnl1

  data_1=data.drop(columns=["Title"],axis=1)
  data_f1=data_1.iloc[len(data_1)-1,]
  label={0.0: 'Neutral',1.0:'Positive',2.0:'Negative'}
  f_data=pd.DataFrame(columns=["Description", "Sentiment"])
  f_data["Description"]=["Rolling 1 day-News Day Event","Rolling 3 day-News Day Event","Rolling 7 day-News Day Event ","Rolling 15 day-News Day Event"]
  f_data["Sentiment"]=[label[data_f1[i]] for i in range(1,len(data_f1),1)]

  # PIE CHART CODE

  labels = ['Neutral', 'Positive','Negative']
  sizes = [data['1day_sentiment'].value_counts()[0],data['1day_sentiment'].value_counts()[1],data['1day_sentiment'].value_counts()[2]]
  # Plot
  fig, ax = plt.subplots(figsize=(4,4))
  #st.subheader('News Classification based on sentiment Analysis')
  ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=140)
  plt.title('News type based on sentiment Analysis')
  buf1 = BytesIO()
  fig.savefig(buf1, format="png")
  
  # Word CLoud
  words=list(data['Title'])
  vocab=Counter()          
  vocab1=[]
  #             
  for i in range(len(words)):
    tokens=words[i].split()
    for j in tokens:
        vocab1.append(j)

  vocab.update(vocab1)
  vocab.most_common(50)

  other_stopwords_to_remove = ['\\n', 'n', '\\', '>', 'nLines', 'nI',"n'", "hi"]
  STOPWORDS = STOPWORDS.union(set(other_stopwords_to_remove))
  stopwords = set(STOPWORDS)
  text = str(vocab)
  wordcloud = WordCloud(width = 1800, height = 1800, background_color ='white', 
                        max_words=200, stopwords = stopwords, min_font_size = 10).generate(text)

  fig, ax = plt.subplots(figsize=(4,4))
  ax.imshow(wordcloud, interpolation='bilinear')
  plt.title("News based word cloud")
  plt.axis("off")
  
  buf2 = BytesIO()
  fig.savefig(buf2, format="png")
  images = [buf1,buf2]
  st.image(images, use_column_width=False)

  st.subheader("Current News Based Sentiment Report")
  st.write("Last News Day Recoded : ",data_f1[0])
  f_data.set_index('Description')
  st.write(f_data)

  st.subheader("Back Testing News Based Investment Oppurtunities")
  st.write("We have back-tested news based investment oppurtunities assuming ***Square off the trade within  next trading day***")
  st.write(data4)
  data4.set_index("Description")
  a=data4['Profit(%)'].idxmax()
  pd.options.display.float_format = '{:,.2f}'.format 
  # Summary dataframe providing details about how the 
  summary=pd.DataFrame(columns=["Description","Event Name","Value"])
  summary['Description']=["Most profitable news event","Most loss incurred for the news event"]
  summary["Event Name"]=[data4["Description"][data4['Profit(%)'].idxmax()],data4["Description"][data4['Loss(%)'].idxmax()]]
  summary["Value"]=[data4['Profit(%)'].max(),data4['Loss(%)'].max()]
  st.write(summary)