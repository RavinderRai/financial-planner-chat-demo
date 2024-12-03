import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY_RAVI"]

client = OpenAI(api_key=openai_api_key)

unique_specialties = [
    'CPP, GIS, etc.', 'Cash flow management', 'Debt Management', 'LGBTQAI2S+ Households', 
    'business owner planning', 'business transition plans', 'corporate planning', 
    'cross-border planning', 'disability planning', 'estate planning', 'expatriate planning', 
    'family business', 'life insurance', 'medical professionals', 'pension plans', 'pro athletes', 
    'rental property', 'retirement income', 'self-employed', 'separation and divorce', 
    'small business', 'small clients', 'tax planning', 'tax prep and filing', 'trusts'
]

# "Select a province..." is meant to be a default option since selectbox in streamlit doesn't have that feature
all_regions = [
    "Select a province...", "Alberta", "British Columbia", "Manitoba", "New Brunswick", 
    "Newfoundland", "Northwest Territories", "Nova Scotia", 
    "Nunavut", "Ontario", "Prince Edward Island", "Quebec", 
    "Saskatchewan", "Yukon"
]

output_parser_response = """You are a financial analyst tasked with evaluating the response from an AI Agent that processes natural language questions to extract information from a financial dataset. Your goal is to refine the output for clarity and conciseness.

If the response contains information about financial planners, rewrite it in nice markdown formatting, ensuring only the top 4 financial planners are included, presented clearly and neatly. If the response indicates that no relevant financial planners were found, rewrite the answer to inform the user that the dataset does not contain any relevant information for their query."""

def filter_provinces(row, selected_region):
    # Split the provinces_served string into a list of provinces
    provinces_list = [province.strip() for province in row.split(',')]
    # Check if the selected region is in the list of provinces
    return selected_region in provinces_list

def filter_specialties(row, selected_specialties):
    # If the row is NaN, return False (to filter out this row)
    if pd.isna(row):
        return False
    # Split the specialties string into a list of specialties
    specialties_list = [specialty.strip() for specialty in row.split(',')]
    # Check if all selected specialties are in the list of specialties
    return all(specialty in specialties_list for specialty in selected_specialties)

def query_dataframe(query, df):
    engine = create_engine("sqlite:///cleaned_demo_data.db")
    
    db = SQLDatabase(engine=engine)
    df.to_sql("cleaned_demo_data", engine, index=False, if_exists='replace')

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    response = agent_executor.invoke({"input": query})

    return response['output']


def main():
    st.set_page_config(page_title="Discover your Financial Planner")

    # font of the app overall seems a bit large, so adding some custom css to reduce it
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-size: 14px;
        }
        .stMarkdown {
            font-size: 12px;
        }
        .stButton > button {
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Financial Planning Assistant:")
    st.subheader("I can refer you to a financial planner. What are you looking for?")

    st.sidebar.title("Select Province or Specialties")
    
    selected_region = st.sidebar.selectbox(
        "Choose your province:", all_regions
    )
    
    selected_specialties = st.sidebar.multiselect(
        "Choose one or more specialties:", 
        unique_specialties,
        placeholder="Select your specialties..."
    )

    df = pd.read_csv('cleaned_demo_data.csv')

    if selected_region and selected_region != "Select a province...":
        filtered_df = df[df['provinces_served'].apply(filter_provinces, selected_region=selected_region)]
    else:
        st.warning("Please select your province to start.")
        st.stop()  # Stop

    # Filter DataFrame by specialties if specialties are selected
    if selected_specialties:
        filtered_df = filtered_df[filtered_df['specialties'].apply(filter_specialties, selected_specialties=selected_specialties)]


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button("Find Top Financial Planners"):
        prompt = f"Who are the top 4 best financial planners in {selected_region} and what is their company, email, specialties and project fees."

        # Get response from PandasAI
        with st.spinner('Processing...'):
            response = query_dataframe(prompt, filtered_df)

        # Send response to OpenAI for finance code review and reformatting
        messages = [
            {"role": "system", "content": output_parser_response},
            {"role": "user", "content": response}
        ]

        openai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            stream=True
        )

        # Display the formatted response
        with st.chat_message("assistant"):
            #st.markdown(final_response)
            final_response = st.write_stream(openai_response)


if __name__ == "__main__":
    main()
