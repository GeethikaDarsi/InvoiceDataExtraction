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


def hash_password(plain_password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode(), salt)

# # Replace 'abc' and 'def' with actual plain text passwords
# hashed_passwords = stauth.Hasher(['abc', 'def']).generate()


# Load user credentials from YAML file
with open("users.yaml", "r") as f:
    users_data = yaml.safe_load(f)
users = {user["username"]: user for user in users_data["users"]}

# Create a login widget
# authenticator.login(location='main')  # You can choose 'main' or 'sidebar'

# Initialize Streamlit-Authenticator
# auth = stauth.Authenticator()


# Authenticate user
def authenticate(username, password):
    if username in users:
        stored_hash = users[username]["password_hash"]
        if users[username]["password_hash"]==password:
            return True
    return False


# Define login widget
def login_widget():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    return username, password

                
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
def main():
    st.title("Invoice Data Extraction")
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

main()