# Core Pkgs
import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 
import altair as alt
from datetime import datetime

# Online ML Pkgs
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords,TFIDF
from river.compose import Pipeline

# Training Data
df = pd.read_csv("data.csv" , delimiter = ',')

data = df.to_records(index = False)



# Model Building
model = Pipeline(('vectorizer',BagOfWords(lowercase=True)),('nv',MultinomialNB()))
for x , y in data:
	model = model.learn_one(x , y)

# Storage in A Database
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Create Fxn From SQL
def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS predictionTable(message TEXT,prediction TEXT,probability NUMBER,Pream_Proba NUMBER,Govern_Proba NUMBER,Indem_Proba NUMBER,LenderD_Proba NUMBER,Other_Proba NUMBER,postdate DATE)')


def add_data(message,prediction,probability,Pream_Proba,Govern_Proba,Indem_Proba,LenderD_Proba,Other_Proba,postdate):
    c.execute('INSERT INTO predictionTable(message,prediction,probability,Pream_Proba,Govern_Proba,Indem_Proba,LenderD_Proba,Other_Proba,postdate) VALUES (?,?,?,?,?,?,?,?,?)',(message,prediction,probability,Pream_Proba,Govern_Proba,Indem_Proba,LenderD_Proba,Other_Proba,postdate))
    conn.commit()

def view_all_data():
	c.execute("SELECT * FROM predictionTable")
	data = c.fetchall()
	return data



def main():
	menu = ["Home","Manage","About"]
	create_table()

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home")
		with st.form(key='mlform'):
			col1,col2 = st.columns([2,1])
			with col1:
				df = st.file_uploader("Upload CSV" , type = ["csv"])
				if df is not None:
					file_details = {"filename":df.name , "filetype": df.type , "filesize": df.size}
					st.write(file_details)
					df = pd.read_csv(df)
					#st.dataframe(df)
				submit_message = st.form_submit_button(label='Predict')

			with col2:
				st.write("Online Incremental ML")
				st.write("Predict Passages as Clauses")

		if submit_message:
			message = df['Passage']
			pred_Arr = []
			for i in message:
				prediction = model.predict_one(i)
				pred_Arr.append(prediction)
				prediction_proba = model.predict_proba_one(i)	
				probability = max(prediction_proba.values())
				postdate = datetime.now()
				# Add Data To Database
				add_data(i,prediction,probability,prediction_proba['Preamble'],prediction_proba['Governing Law'],prediction_proba['Indemnification'],prediction_proba['Lender Defaulting'],prediction_proba['Other'],postdate)
				st.success("Data Submitted")


			res_col1 ,res_col2 = st.columns(2)
			with res_col1:
				st.info("Original Passage")
				st.text(df.to_records())

			with res_col2:
				st.info("Prediction")
				st.write(pred_Arr)

				# Plot of Probability
				#df_proba = pd.DataFrame({'label':prediction_proba.keys(),'probability':prediction_proba.values()})
				# st.dataframe(df_proba)
				# visualization
				#fig = alt.Chart(df_proba).mark_bar().encode(x='label',y='probability')
				#st.altair_chart(fig,use_container_width=True)	





	elif choice == "Manage":
		st.subheader("Manage & Monitor Results")
		stored_data =  view_all_data() 
		new_df = pd.DataFrame(stored_data,columns=['message','prediction','probability','Pream_Proba','Govern_Proba','Indem_Proba','LenderD_Proba','Other_Proba','postdate'])
		st.dataframe(new_df)
		new_df['postdate'] = pd.to_datetime(new_df['postdate'])

		# c = alt.Chart(new_df).mark_line().encode(x='minutes(postdate)',y='probability')# For Minutes
		c = alt.Chart(new_df).mark_line().encode(x='postdate',y='probability')
		st.altair_chart(c)
	
		c_Pream_Proba = alt.Chart(new_df['Pream_Proba'].reset_index()).mark_line().encode(x='Pream_Proba',y='index')
		c_Govern_Proba = alt.Chart(new_df['Govern_Proba'].reset_index()).mark_line().encode(x='Govern_Proba',y='index')
		c_Indem_Proba = alt.Chart(new_df['Indem_Proba'].reset_index()).mark_line().encode(x='Indem_Proba',y='index')
		c_LenderD_Proba = alt.Chart(new_df['LenderD_Proba'].reset_index()).mark_line().encode(x='LenderD_Proba',y='index')
		c_Other_Proba = alt.Chart(new_df['Other_Proba'].reset_index()).mark_line().encode(x='Other_Proba',y='index')
		

		
		with st.expander("Preamble Probability"):
			st.altair_chart(c_Pream_Proba,use_container_width=True)

		with st.expander("Governing Probability"):
			st.altair_chart(c_Govern_Proba,use_container_width=True)

		with st.expander("Indemnification Probability"):
			st.altair_chart(c_Indem_Proba,use_container_width=True)

		with st.expander("Lender Defaulting Probability"):
			st.altair_chart(c_LenderD_Proba,use_container_width=True)

		with st.expander("Other Probability"):
			st.altair_chart(c_Other_Proba,use_container_width=True)

	else:
		st.subheader("About")
		st.write()



if __name__ == '__main__':
	main()


