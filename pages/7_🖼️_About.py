import streamlit as st

st.set_page_config(page_title="About", page_icon='🖼️')

st.markdown("<h1 style='text-align: center;'>🖼️ About this App</h1>", unsafe_allow_html=True)

st.markdown("---")

st.markdown(""" This app was designed and developed by Group 5 - IE MBD 2023/2024.

The team members are:
* Jorge García
* Vanessa Vega
* Mario Bevilacqua
* Andrés Malvido
* Brennan Gordon       

**November 2023**""")

st.markdown("---")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)