import streamlit as st
import google.generativeai as genai
import time
import streamlit_shadcn_ui as ui

# Set up the Streamlit app configuration
st.set_page_config(
    page_title="Navy Chat - Low Code Tier",
    page_icon="🚢",
    layout="centered"
)




# Sidebar configuration
with st.sidebar:
    st.title("🚢 Medical Navy Chatbot 💬")
    st.write("Configure your chatbot settings below.")

    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.slider("Max Output Tokens", 500, 8192, 8192)
    topk = st.slider("top_k", 5, 100, 64)
    topp = st.slider("top_p", 0.5, 1.0, 0.8)


pdf_paths = [
    '../docs/US_Mercy.pdf',
    '../docs/A_Decade_of_Surgery_Aboard_the_U.S._COMFORT.pdf',
    '../docs/Hospital_ships_adrift__Part_2__The_role_of_U.S._Navy_hospital_ship.pdf',
    '../docs/Sea_Power__The_U.S._Navy_and_Foreign_Policy__Council_on_Foreign_Relations.pdf',
    '../docs/US_Navy_Ship-Based_Disaster_Response__Lessons_Learned_-_PMC.pdf'
]


# Main Streamlit app
st.title("💬 Chatbot - Low Code Tier")
st.caption("This chatbot is trained on the following PDFs shown below.")

ui.badges(badge_list=[
    ("Humanitarian Assistance and Disaster Relief Aboard the USNS Mercy", "secondary"),
    ("US Navy Ship-Based Disaster Response", "secondary"),
    ("Sea Power: The US Navy and Foreign Policy", "secondary"),
    ("A Decade of Surgery Abroad the US Naval Ship Comfort", "secondary"),
    ("Hospital Ships Adrift?", "secondary"),
], class_name="flex gap-2", key="badges1")

# Initialize Low Code Tier chat history in session state
if "low_code_chat_history" not in st.session_state:
    st.session_state.low_code_chat_history = []

# Display chat history with UI effect for generating response
for msg in st.session_state.low_code_chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if "prev_temp" not in st.session_state:
    st.session_state.prev_temp = temp
if "prev_max_tokens" not in st.session_state:
    st.session_state.prev_max_tokens = max_tokens
if "prev_topk" not in st.session_state:
    st.session_state.prev_topk = topk
if "prev_topp" not in st.session_state:
    st.session_state.prev_topp = topp

# Global model variable initialized as None
model = None

def initialize(tempurature=0.2, toks=8192, top_k=64, top_p=0.8):
    with st.spinner("Updating model..."):
    
        # Configure Google Gemini AI
        genai.configure(api_key="AIzaSyCxkUFNQKbkfcE0Mj1EDmzNfp7Tr99CAsU")
        generation_config = {
            "temperature": tempurature,
            "max_output_tokens": toks,
            "top_k": top_k,
            "top_p": top_p,
        }

        # Initialize the generative model
        global model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
        )

        
        # Upload PDFs only if model is initialized successfully
        #uploaded=[genai.upload_file(path=path) for path in pdf_paths]



# Ensure the model is initialized only once, with user-defined parameters if provided
# Check for slider changes and initialize if any change detected
if model is None or (temp != st.session_state.prev_temp or max_tokens != st.session_state.prev_max_tokens or topk != st.session_state.prev_topk or topp != st.session_state.prev_topp):
    
    # Update previous values in session state
    st.session_state.prev_temp = temp
    st.session_state.prev_max_tokens = max_tokens
    st.session_state.prev_topk = topk
    st.session_state.prev_topp = topp

    # Run model initialization
    initialize(tempurature=temp, toks=max_tokens, top_k=topk, top_p=topp)


# Function to generate response using Google Gemini model
def get_ai_response_with_loading(question):
    try:
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "Sorry, I couldn't process that request."

# Capture user input
user_question = st.chat_input("Enter your question:")
if user_question:
    st.session_state.low_code_chat_history.append({"role": "user", "content": user_question})
    
    # Display user question
    with st.chat_message("user"):
        st.write(user_question)
    
    # Generate and display assistant's response progressively
    with st.chat_message("assistant"):
        placeholder = st.empty()  # Placeholder for progressive updates

        # Loading effect displayed initially
        loading_html = """
        <style>
        .loading-message {
            animation: fadeInOut 3s ease-in-out infinite;
        }
        @keyframes fadeInOut {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }
        </style>
        <div class="loading-message">Generating response...</div>
        """
        placeholder.markdown(loading_html, unsafe_allow_html=True)

        # Short delay to simulate loading before response starts displaying
        time.sleep(1)

        # Fetch AI response
        answer_text = get_ai_response_with_loading(user_question)
        
        # Progressive display of the response text
        placeholder.empty()  # Clear loading message
        displayed_text = ""
        for char in answer_text:
            displayed_text += char
            placeholder.write(displayed_text)
            time.sleep(0.003)  # Lower delay for faster rendering
        
        # Add completed answer to chat history
        st.session_state.low_code_chat_history.append({"role": "assistant", "content": displayed_text})