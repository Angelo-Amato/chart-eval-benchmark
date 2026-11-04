import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from st_supabase_connection import SupabaseConnection

# --- CONFIGURATION ---
DATA_PATH = "human_eval_set.csv"
IMAGE_DIR = "data/raw"
RENDER_BUFFER = 0.5  # Seconds to wait for image to load before strictly counting down
TIME_LIMIT = 10      # Locked 10-second limit for Quick mode

st.set_page_config(page_title="Chart Comparison Human Eval", layout="centered")

# --- FEEDBACK FORM (SIDEBAR) ---
# Putting this at the top level ensures it is accessible on every screen
with st.sidebar:
    st.header("📝 Leave Feedback")
    st.write("Find a bug or have a suggestion? Let us know!")
    
    with st.form("feedback_form", clear_on_submit=True):
        fb_name = st.text_input("Name/Initials (optional)")
        # Removed required=True to prevent Streamlit TypeError
        fb_text = st.text_area("Your feedback...") 
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            # Manually check if the feedback text is empty
            if not fb_text.strip():
                st.error("Please enter some feedback before submitting.")
            else:
                with st.spinner("Sending..."):
                    try:
                        conn = st.connection("supabase", type=SupabaseConnection)
                        conn.table("app_feedback").insert({
                            "user_name": str(fb_name),
                            "feedback_text": str(fb_text)
                        }, count="None").execute()
                        st.success("Feedback sent! Thank you.")
                    except Exception as e:
                        st.error(f"Error saving feedback: {e}")

# --- PARSE URL FOR TWO DIFFERENT LINKS ---
if "mode" in st.query_params:
    target_mode = st.query_params["mode"].capitalize()
    if target_mode not in ["Quick", "Reasoning"]:
        target_mode = "Reasoning" # Default if misspelled
else:
    target_mode = "Reasoning" # Default if no URL parameter is provided

# --- INITIALIZE SESSION STATE ---
if 'q_in_batch' not in st.session_state:
    st.session_state.q_in_batch = 0
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'app_state' not in st.session_state:
    st.session_state.app_state = "SETUP" 
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'batch_df' not in st.session_state:
    st.session_state.batch_df = None
if 'batch_num' not in st.session_state:
    st.session_state.batch_num = 1

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Cache the image reading so we don't hit the hard drive repeatedly
@st.cache_data
def load_image_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

df = load_data()

# Calculate maximum possible batches based on having 5 Hard and 5 Easy per batch
hard_df = df[df['machine_difficulty'] == 'Hard']
easy_df = df[df['machine_difficulty'] == 'Easy']
max_batches = min(len(hard_df) // 5, len(easy_df) // 5)
if max_batches < 1: 
    max_batches = 1

def get_batch_data(batch_number):
    start_idx = (batch_number - 1) * 5
    end_idx = batch_number * 5
    
    batch_hard = hard_df.iloc[start_idx:end_idx]
    batch_easy = easy_df.iloc[start_idx:end_idx]
    
    # Combine the 5 hard and 5 easy, then shuffle the row order
    combined = pd.concat([batch_hard, batch_easy])
    return combined.sample(frac=1).reset_index(drop=True)

def record_answer(answer):
    current_row = st.session_state.batch_df.iloc[st.session_state.q_in_batch]
    
    if answer == "unanswered":
        is_correct = False
    else:
        human_pred_diff = (answer == "Different")
        is_correct = (human_pred_diff == current_row['actual_diff'])
    
    st.session_state.responses.append({
        'id': current_row['id'],
        'human_answer': answer,
        'human_correct': is_correct,
        'machine_difficulty': current_row['machine_difficulty'],
        'eval_mode': target_mode,
        'batch_num': st.session_state.batch_num
    })
    
    st.session_state.q_in_batch += 1
    
    # Check if batch is complete (10 questions)
    if st.session_state.q_in_batch >= len(st.session_state.batch_df):
        st.session_state.app_state = "RESULTS"
    else:
        st.session_state.start_time = time.time() + RENDER_BUFFER

# --- 1. SETUP SCREEN ---
if st.session_state.app_state == "SETUP":
    st.title("Human Evaluation Setup")
    
    st.markdown(f"### Current Mode: **{target_mode}**")
    if target_mode == "Quick":
        st.info(f"You will have a maximum of {TIME_LIMIT} seconds to evaluate each chart.")
    else:
        st.info("You may take as much time as you need to analyze each chart.")
        
    st.markdown("---")
    st.write("Please select your assigned batch number:")
    
    selected_batch = st.number_input("Batch Number", min_value=1, max_value=max_batches, value=1)
    
    if st.button("Start Evaluation", use_container_width=True):
        st.session_state.batch_num = selected_batch
        st.session_state.batch_df = get_batch_data(selected_batch)
        
        # --- CACHE PRE-WARMING LOOP ---
        with st.spinner("Pre-loading charts to ensure accurate timing..."):
            for _, row in st.session_state.batch_df.iterrows():
                img_path = os.path.join(IMAGE_DIR, f"{row['id']}.png")
                if os.path.exists(img_path):
                    # Calling this here forces Streamlit to save the bytes to RAM
                    load_image_bytes(img_path) 
                    
        st.session_state.app_state = "QUIZ"
        st.session_state.start_time = time.time() + RENDER_BUFFER
        st.rerun()

# --- 2. QUIZ INTERFACE ---
elif st.session_state.app_state == "QUIZ":
    
    is_quick_mode = (target_mode == "Quick")
    total_in_batch = len(st.session_state.batch_df)