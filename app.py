import pandas as pd
import numpy as np
import streamlit as st

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
