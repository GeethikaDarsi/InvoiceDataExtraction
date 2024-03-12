import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sqlite3
import bcrypt
import streamlit as st
import cv2
import easyocr
import re
from ultralytics import YOLO
import pandas as pd
from fuzzywuzzy import fuzz
import imghdr
import warnings
import csv
import ast
import yaml
from yaml import SafeLoader
import bcrypt
import streamlit_authenticator as stauth
import sqlite3


                
model = YOLO('best.pt')

def clean_string(input_list):
		combined_string = ' '.join(input_list)
		cleaned_string = re.sub(r'[^\w\s]', '', combined_string)
		return cleaned_string
	
def fuzzy_match(template_text, extracted_text):
	return fuzz.token_set_ratio(template_text, extracted_text)
		
def load_csv_data(csv_path):
	return pd.read_csv(csv_path)

def store_in_database(class_texts, database_file='database.csv'):
	# Read the existing database file into a pandas DataFrame
	df = pd.read_csv(database_file)

		# Convert the class_texts dictionary to a DataFrame
	new_data = pd.DataFrame([class_texts])

		# Append the new data
	df = pd.concat([df, new_data], ignore_index=True)

		# Write the DataFrame back to the CSV file
	df.to_csv(database_file, index=False)

# Main app
def main_1():
    input_file_name = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]).name
    # if input_file_name:
    results = model.predict(source=input_file_name, show=True, save=True, conf=0.8, line_thickness=2, save_crop=True)
    csv_path = "database.csv"
    orig_img = results[0].orig_img
    reader = easyocr.Reader(['en'])
    boxes = results[0].boxes.xyxy 
    class_texts = {}
    boxes = results[0].boxes.xyxy
    text_list=[]
    class_texts = {}
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped_img = orig_img[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        text = reader.readtext(gray, detail=0)
        text = clean_string(text)
        text_list.append(text)
        class_name = results[0].names[i]
        class_texts[class_name] = text
        # st.write(class_texts)
    store_in_database(class_texts)
    st.write("Extracted data is stored in the database succesfully")


# Initialize SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def signup(username, password):
    # Check if user already exists
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    if c.fetchone():
        return False
    else:
        # Hash the password and store the user
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return True

def login(username, password):
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if user and verify_password(password, user[1]):
        return True
    else:
        return False




if __name__=='__main__':
    st.title("Invoice Data Extraction")


    auth_status = st.session_state.get('auth_status', None)
    if auth_status == "logged_in":
        st.success(f"Welcome {st.session_state.username}!")
        main_1()


    elif auth_status == "login_failed":
        st.error("Login failed. Please check your username and password.")
        auth_status = None
    elif auth_status == "signup_failed":
        st.error("Signup failed. Username already exists.")
        auth_status = None
    # Login/Signup form
    if auth_status is None or auth_status == "logged_out":
        form_type = st.radio("Choose form type:", ["Login", "Signup"])

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if form_type == "Login":
            if st.button("Login"):
                if login(username, password):
                    st.session_state.auth_status = "logged_in"
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.session_state.auth_status = "login_failed"
                    st.rerun()
        else:  # Signup
            if st.button("Signup"):
                if signup(username, password):
                    st.session_state.auth_status = "logged_in"
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.session_state.auth_status = "signup_failed"
                    st.rerun()

    # Logout button
    if auth_status == "logged_in":
        if st.button("Logout"):
            st.session_state.auth_status = "logged_out"
            del st.session_state.username
            st.rerun()
