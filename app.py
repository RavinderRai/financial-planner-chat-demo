import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai import SmartDataframe
from dotenv import load_dotenv

load_dotenv()

#openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

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

    llm = PandasOpenAI(api_token=openai_api_key)
    query_engine = SmartDataframe(
        filtered_df,
        config={
            "llm": llm,
        },
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check if any region is selected before accepting user input
    if prompt := st.chat_input("Enter your question about financial planners or services."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from PandasAI
        with st.spinner('Processing...'):
            pandasai_response = query_engine.chat(prompt)

        # Convert PandasAI response to string
        if isinstance(pandasai_response, pd.DataFrame):
            # If it's a DataFrame, convert to a readable string
            pandasai_response_str = pandasai_response.to_string(index=False)
        else:
            # Ensure the response is always a string
            pandasai_response_str = str(pandasai_response)

        # Send response to OpenAI for finance code review and reformatting
        messages = [
            {"role": "system", "content": f"You are a finance code reviewer. You will be given an output from PandasAI in response to a natural language question about a financial dataset. If the answer is showing a set of results, only list the top 4, in nice markdown formatting. And if not, then just rewrite it in nice markdown formatting. Again, ensure that the final text is nicely formatted with proper spacing and no italicized text."},
            {"role": "user", "content": pandasai_response_str}
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
            response = st.write_stream(openai_response)

        # Save messages in session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
