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
RENDER_BUFFER = 2.0  # Seconds to wait for image to load before strictly counting down
TIME_LIMIT = 10      # Locked 10-second limit for Quick mode

st.set_page_config(page_title="Chart Comparison Human Eval", layout="centered")

# --- PARSE URL FOR TWO DIFFERENT LINKS ---
# This reads ?mode=Quick or ?mode=Reasoning from the URL
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
    
    if is_quick_mode:
        st_autorefresh(interval=1000, key=f"timer_{st.session_state.q_in_batch}")
        
    st.title(f"Evaluation: {target_mode} Mode")
    st.progress(st.session_state.q_in_batch / total_in_batch)
    st.write(f"**Batch {st.session_state.batch_num}** | Question {st.session_state.q_in_batch + 1} of {total_in_batch}")
    
    # Timer Logic
    if is_quick_mode:
        elapsed = time.time() - st.session_state.start_time
        remaining = min(TIME_LIMIT, int(TIME_LIMIT - elapsed))
        
        if remaining <= 0:
            record_answer("unanswered")
            st.rerun()
        else:
            st.warning(f"⏱️ Time Remaining: **{remaining}s**")

    current_row = st.session_state.batch_df.iloc[st.session_state.q_in_batch]
    sample_id = current_row['id']
    
    img_path = os.path.join(IMAGE_DIR, f"{sample_id}.png")
    
    if os.path.exists(img_path):
        img_bytes = load_image_bytes(img_path)
        st.image(img_bytes, use_container_width=True)
    else:
        st.error(f"Image not found at: {img_path}")
    
    st.markdown("### Are these two charts exactly the same, or are there informational differences?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("They are the SAME", key=f"same_{st.session_state.q_in_batch}", on_click=record_answer, args=("Same",), use_container_width=True)
    with col2:
        st.button("They are DIFFERENT", key=f"diff_{st.session_state.q_in_batch}", on_click=record_answer, args=("Different",), use_container_width=True)

# --- 3. RESULTS DASHBOARD ---
elif st.session_state.app_state == "RESULTS":
    st.title("Evaluation Complete! 📊")
    st.balloons()
    
    results_df = pd.DataFrame(st.session_state.responses)
    
    if not results_df.empty:
        overall_acc = results_df['human_correct'].mean() * 100
        unanswered_count = len(results_df[results_df['human_answer'] == 'unanswered'])
        
        st.markdown("### Your Performance")
        col1, col2 = st.columns(2)
        col1.metric("Overall Score", f"{overall_acc:.1f}%")
        col2.metric("Timeouts", unanswered_count)
        
        st.markdown("---")
        st.markdown("### Save Your Score")
        user_initials = st.text_input("Name/Initials:", placeholder="e.g., John Smith")
        
        if 'data_saved' not in st.session_state:
            st.session_state.data_saved = False
            
        if user_initials and not st.session_state.data_saved:
            if st.button("Submit Results to Database", use_container_width=True):
                with st.spinner("Saving to database..."):
                    try:
                        conn = st.connection("supabase", type=SupabaseConnection)
                        
                        db_payload = []
                        for response in st.session_state.responses:
                            db_payload.append({
                                "evaluator_name": str(user_initials),
                                "sample_id": str(response['id']),
                                "human_answer": str(response['human_answer']),
                                "human_correct": bool(response['human_correct']), # explicitly cast to pure Python bool
                                "machine_difficulty": str(response['machine_difficulty']),
                                "eval_mode": str(response['eval_mode']),
                                "batch_num": int(response['batch_num'])
                            })
                            
                        conn.table("evaluation_results").insert(db_payload, count="None").execute()
                        
                        st.session_state.data_saved = True
                        st.success("Results successfully saved to the database! You may close this tab.")
                        st.rerun() 
                        
                    except Exception as e:
                        st.error(f"Error saving to database: {e}")
                        
        elif st.session_state.data_saved:
            st.success(f"Thank you, {user_initials}! Your results are safely stored.")
        else:
            st.info("Please enter your name or initials to unlock the submit button.")
    else:
        st.warning("No data recorded.")