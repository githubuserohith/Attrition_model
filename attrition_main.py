import attrition_model as am
import attrition_mlflow as al
import attrition_streamlit as ast
import os
import pandas as pd
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

# page_bg_img = '''
# <style>
# body {
# background-image: url("https://imgur.com/aqgmhvG");
# background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)
# # os.chdir(r"")

df = pd.read_csv("IBM.csv")

model,pp,X_train,X_test,y_train,y_test,model_list = am.fn_model(df)
# print(model)
ast.fn_st(model,df,pp)
al.fn_mlflow(model,X_train,X_test,y_train,y_test,model_list)




