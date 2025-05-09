import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
import os

#

# Setup streamlit frontend interface
st.set_page_config(page_title ="Text to math problem solver and data search assistant", page_icon = "📚" )
st.title("📚 Text to math problem solver using Google Gemma2 LLM Model")

groq_api_key = st.sidebar.text_input(label = "Groq API key", type = "password")

if not groq_api_key:
    st.info("Please provide the Groq-API key to continue")
    st.stop()
    
llm = ChatGroq(model = "gemma2-9b-it", api_key = groq_api_key)

# Initializing wikipedia tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = "Wikipedia", 
    func = wikipedia_wrapper.run, 
    description = "A tool for searching the internet to find various information on the topics mentioned")

# Initialize math tool
math_chain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name = "Calculator",
    func = math_chain.run,
    description = "A tool for answering only math related question. Only input mathematical expression needs to be provided")

# Initializing prompt 
prompt = """ 
    You are a agent tasked to solve mathematical question. Logically arrive at the solution
    and provide a detailed explanation and display it point whise for the question below
    Question : {question}
"""

prompt_template = PromptTemplate(
    input_variables = ['question'],
    template = prompt 
)

# Combining all the tools into chain 
chain = LLMChain(llm = llm, prompt = prompt_template)

# Creating reasoning tool            it is interacting with the llm model have some thought of chain
reasoning_tool = Tool(
    name = "reasoning",
    func = chain.run,
    description = "A tool for answering Logic based and reasoning questions"
)

# initializing agent
assistant_agents = initialize_agent(
    tools = [wikipedia_tool, calculator, reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    handle_parsing_errors = True
)

# For chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role" : "assistant", 
                                     "content" : "Hi I'm a math chat bot who can answer all the math related questions"}]
    
for i in st.session_state.messages:
    st.chat_message(i["role"]).write(i["content"])

# Lets start the interaction
question = st.text_area("Enter your question")

if st.button("Find my answer: "):
    if question:
        with st.spinner("Generating Response..."):
            st.session_state.messages.append({"role" : "user", "content" : question})
            st.chat_message("user").write(question)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = True) 
            response = assistant_agents.run(st.session_state.messages, callbacks = [st_cb])
            st.session_state.messages.append({"role" : "assistant", "content" : response})
            st.write("Response: ")
            st.success(response) 
    else:
        st.warning("Please enter your question")           
            
