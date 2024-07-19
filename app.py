import streamlit as st
from nltk.corpus import stopwords
import re
import pandas as pd
import os
import joblib
from src.logger import logging



st.title('Zomato Rating Prediction..')


def preprocess_dish(a : str):
    stop_words = set(stopwords.words('english'))
    words = re.sub(r'[^a-zA-Z]', ' ', a)
    words = words.split()
    words = [i.strip() for i in words]
    filtered_text = ' '.join([word for word in words if word.lower() not in stop_words])
    return filtered_text


def preprocess_cusin(x : str):
    x = x.lower()
    x = x.split(',')
    x = [i.strip() for i in x]
    x = sorted(x)
    return ' '.join(x)


current_dir = os.getcwd()
model_path = os.path.join(current_dir, "models/model.pkl")
preprocessor_path = os.path.join(current_dir, "models/preprocessor.pkl")


datas = pd.read_csv('data/zomato.csv')
nums = ['votes','approx_cost(for two people)']
cat = ['online_order', 'book_table', 'rest_type',  'listed_in(type)', 'listed_in(city)']
tex = ['cuisines']



lis = list()
s = st.number_input('Enter the no of votes given to the zomato ',min_value=datas['votes'].min(),max_value=datas['votes'].max(),value=int(datas['votes'].median()),step=1)
s1 = st.number_input('Enter the Approximate cost for the two people',min_value=datas['approx_cost(for two people)'].min(),max_value=datas['approx_cost(for two people)'].max(),value=datas['approx_cost(for two people)'].median(),step=10.55)
lis.append([s])
lis.append([s1])
for i  in cat:
    a = st.selectbox(i.title(), options=datas[i].unique())
    lis.append([a])
# user_input0 = st.text_area("Enter the dish liked by the people .")
user_input1 = st.text_area("Enter the cuisined had by zomato  restuarent.")
# user_input0 = preprocess_dish(user_input0)
user_input1 = preprocess_cusin(user_input1)
# lis.append([user_input0])
lis.append([user_input1])


@st.cache_data
def load_model(preprocessor_path):
    try:
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Model file '{preprocessor_path}' not found.")
        
        preprocessor = joblib.load(preprocessor_path)
        logging.info(f"Loaded preprocessor from {preprocessor_path}")
        return preprocessor
    except FileNotFoundError:
        logging.error(f"File '{preprocessor_path}' not found.")
        return None
    except Exception as e:
        logging.error(f"Error loading preprocessor: {str(e)}")
        return None




if st.button('Predict Rating :'):
    cols = nums + cat + tex
    print("Columns : ",cols)
    df = pd.DataFrame(lis).T
    df.columns = cols
    print("Data Frame: ",df.head())
    model = load_model(model_path)
    preprocessor = load_model(preprocessor_path)
    df = preprocessor.transform(df)
    prediction = model.predict(df)
    st.write(f"Prediction of the rating of the zomato restuarent : {prediction[0]:.2f} * Rating has been given .")





