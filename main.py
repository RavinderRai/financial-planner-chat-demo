import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks import OpenAICallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Custom callback handler for Streamlit streaming
from langchain.callbacks.base import BaseCallbackHandler

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.content = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.content += token
        self.placeholder.markdown(self.content)

def main():
    st.set_page_config(page_title="Chat with your Financial Data")
    st.header("Financial Planner Recommendation")
    st.subheader("I can refer you to a financial planner. What are you looking for?")


    # GPT Model selection
    model = "gpt-4o-mini"
    api_key = os.environ["OPENAI_API_KEY"]

    engine = create_engine("sqlite:///financial_demo_data.db")
    db = SQLDatabase(engine=engine)

    # Initialize LLM with user-provided API key and streaming enabled
    llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key, streaming=True)

    # Create SQL agent with streaming callback
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    # User input for querying
    user_question = st.text_input(
        "Ask a question about your data:", 
        placeholder="e.g., Who are the financial planners in Alberta?"
    )

    if user_question:
        with st.spinner(text="Processing your request..."):
            response_placeholder = st.empty()

            # Create the custom callback handler for streaming
            streamlit_callback_handler = StreamlitCallbackHandler(response_placeholder)

            # Execute the SQL agent with streaming callback
            try:
                # Run the agent and stream results
                response = agent_executor.invoke(
                    {"input": user_question},
                    callbacks=[streamlit_callback_handler]
                )
            except Exception as e:
                response_placeholder.error(f"Error: {e}")

            # Final output
            response_placeholder.markdown(response)

if __name__ == "__main__":
    main()
