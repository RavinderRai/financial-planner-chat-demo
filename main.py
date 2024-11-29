import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

def main():
    st.set_page_config(page_title="Chat with your Financial Data")
    st.header("Chat with your Financial Data 📈")

    # API key input
    st.sidebar.title("API Settings")
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

    if not api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

    # GPT Model selection
    model = st.sidebar.selectbox(
        "Select GPT Model:",
        options=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4"],
        index=0
    )

    # Load database
    df = pd.read_csv('financial_demo_data.csv')
    engine = create_engine("sqlite:///financial_demo_data.db")
    db = SQLDatabase(engine=engine)

    # Load the dataset preview
    if st.checkbox("Show Data Preview"):
        st.dataframe(df.head())


    # Initialize LLM with user-provided API key
    llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    # User input for querying
    user_question = st.text_input(
        "Ask a question about your data:", 
        placeholder="e.g., Who are the financial planners in Alberta?"
    )

    if user_question:
        with st.spinner(text="In progress..."):
            response = agent_executor.invoke({"input": user_question})
            st.write(response['output'])

if __name__ == "__main__":
    main()
