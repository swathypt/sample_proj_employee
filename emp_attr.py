from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import datetime

model = load_model('trained_model')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('employeeleftimage.jpg')
    image_office = Image.open('office.jpg')
    st.sidebar.date_input('Date',datetime.date(2021,7,24))
    
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Form", "Batch"))
    st.sidebar.info('This app is created to predict if an employee will leave the company')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)
    st.title("Predicting employee leaving")
    if add_selectbox == 'Form':
        satisfaction_level=st.number_input('satisfaction_level' , min_value=0.1, max_value=1.0, value=0.1)
        last_evaluation =st.number_input('last_evaluation',min_value=0.1, max_value=1.0, value=0.1)
        number_project = st.number_input('number_project', min_value=0, max_value=50, value=5)
        time_spend_company = st.number_input('time_spend_company', min_value=1, max_value=10, value=3)
        Work_accident = st.number_input('Work_accident',  min_value=0, max_value=50, value=0)
        promotion_last_5years = st.number_input('promotion_last_5years',  min_value=0, max_value=50, value=0)
        salary = st.selectbox('salary', ['low', 'high','medium'])
        department = st.selectbox('department', ['sales','accounting','hr','technical','support','management','IT','product_mng','marketing','RandD'])
        average_montly_hours =st.number_input('average_montly_hours',min_value=1, max_value=1000, value=1)
        output=""
        input_dict={'satisfaction_level':satisfaction_level,'last_evaluation':last_evaluation,'number_project':number_project,'time_spend_company':time_spend_company,'Work_accident': Work_accident,'promotion_last_5years':promotion_last_5years,'salary' : salary,'department':department,'average_montly_hours':average_montly_hours}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            if output == 1:
                output = "leave the company :( "
            else:
                output = "continue :)"
            #output = str(output)
        st.success('Employee will {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)            
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
