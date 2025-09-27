import streamlit as st


#Set initial Page Config
st.set_page_config("ReturnRed", layout="wide", initial_sidebar_state="collapsed")


#st.markdown("<div style='text-align: center;'><h1>Welcome to ReturnRed</h1></div>", unsafe_allow_html=True)


#Generate tabs for each part of the site
dashTab, graphTab, aiTab = st.tabs(["Dashboard", "Visuallize", "Predict"])


with st.container():
   column1, column2 = st.columns(2)

   with column1:
         st.header("Grid 1")
         st.text("THIS WILL BE THE HOME PAGE")

      #Allocate Content under Each tab
         st.header("Graph")  
         st.text("THIS WILL BE THE Graph(ig) PAGE")

   with column2:
         st.header("AI")
         st.text("AI part of the project goes here")

