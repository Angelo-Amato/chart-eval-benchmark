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

# --- INITIALIZE SESSION STATE ---
if 'global_idx' not in st.session_state:
    st.session_state.global_idx = 0
if 'q_in_batch' not in st.session_state:
    st.session_state.q_in_batch = 0
if 'batch_num' not in st.session_state:
    st.session_state.batch_num = 1
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'app_state' not in st.session_state:
    st.session_state.app_state = "SETUP" # States: SETUP, QUIZ, INTERMISSION, RESULTS
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_image_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

df = load_data()
total_questions = len(df)

def record_answer(answer):
    current_row = df.iloc[st.session_state.global_idx]
    
    # Check if human is correct
    if answer == "unanswered":
        is_correct = False
    else:
        human_pred_diff = (answer == "Different")
        is_correct = (human_pred_diff == current_row['actual_diff'])
    
    # Tag which mode they were in for this specific answer
    current_mode = "Reasoning" if st.session_state.q_in_batch < 5 else "Quick"
    
    st.session_state.responses.append({
        'id': current_row['id'],
        'human_answer': answer,
        'human_correct': is_correct,
        'machine_difficulty': current_row['machine_difficulty'],
        'eval_mode': current_mode,
        'batch_num': st.session_state.batch_num
    })
    
    # Advance counters
    st.session_state.global_idx += 1
    st.session_state.q_in_batch += 1
    
    # Determine next state
    if st.session_state.global_idx >= total_questions:
        st.session_state.app_state = "RESULTS"
    elif st.session_state.q_in_batch >= 10:
        st.session_state.app_state = "INTERMISSION"
    else:
        # Just move to next question
        st.session_state.start_time = time.time() + RENDER_BUFFER

# --- 1. SETUP SCREEN ---
if st.session_state.app_state == "SETUP":
    st.title("Human Evaluation Setup")
    st.markdown("""
    **Instructions:**
    This evaluation is split into batches of 10 charts. 
    * **First 5 charts:** *Reasoning Mode* (Take as much time as you need).
    * **Next 5 charts:** *Quick Mode* (Locked to 10 seconds per chart).
    
    After each batch of 10, you can choose to continue or save your results.
    """)
    
    if st.button("Start Batch 1", use_container_width=True):
        st.session_state.app_state = "QUIZ"
        st.session_state.start_time = time.time() + RENDER_BUFFER
        st.rerun()

# --- 2. QUIZ INTERFACE ---
elif st.session_state.app_state == "QUIZ":
    
    is_quick_mode = (st.session_state.q_in_batch >= 5)
    current_mode_label = "Quick (10s limit)" if is_quick_mode else "Reasoning (Untimed)"
    
    # Trigger background auto-refresh every 1000ms ONLY in Quick mode
    if is_quick_mode:
        st_autorefresh(interval=1000, key=f"timer_{st.session_state.global_idx}")
        
    st.title("Human Evaluation: Chart Comparison")
    st.progress(st.session_state.global_idx / total_questions)
    
    st.write(f"**Batch {st.session_state.batch_num}** | Question {st.session_state.q_in_batch + 1} of 10")
    st.write(f"Current Mode: **{current_mode_label}**")
    
    # Timer Logic
    if is_quick_mode:
        elapsed = time.time() - st.session_state.start_time
        remaining = min(TIME_LIMIT, int(TIME_LIMIT - elapsed))
        
        if remaining <= 0:
            record_answer("unanswered")
            st.rerun()
        else:
            st.warning(f"⏱️ Time Remaining: **{remaining}s**")

    current_row = df.iloc[st.session_state.global_idx]
    sample_id = current_row['id']
    
    # Load image optimally
    img_path = os.path.join(IMAGE_DIR, f"{sample_id}.png")
    
    if os.path.exists(img_path):
        img_bytes = load_image_bytes(img_path)
        st.image(img_bytes, use_container_width=True)
    else:
        st.error(f"Image not found at: {img_path}")
    
    st.markdown("### Are these two charts exactly the same, or are there informational differences?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("They are the SAME", key=f"same_{st.session_state.global_idx}", on_click=record_answer, args=("Same",), use_container_width=True)
    with col2:
        st.button("They are DIFFERENT", key=f"diff_{st.session_state.global_idx}", on_click=record_answer, args=("Different",), use_container_width=True)

# --- 3. INTERMISSION SCREEN ---
elif st.session_state.app_state == "INTERMISSION":
    st.title(f"Batch {st.session_state.batch_num} Complete! 🎉")
    st.write(f"You've completed {st.session_state.global_idx} charts so far.")
    st.write("Would you like to complete another set of 10, or stop here and save your data?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continue to Next Batch", use_container_width=True):
            st.session_state.batch_num += 1
            st.session_state.q_in_batch = 0
            st.session_state.app_state = "QUIZ"
            st.session_state.start_time = time.time() + RENDER_BUFFER
            st.rerun()
    with col2:
        if st.button("Finish and Save", use_container_width=True):
            st.session_state.app_state = "RESULTS"
            st.rerun()

# --- 4. RESULTS DASHBOARD ---
elif st.session_state.app_state == "RESULTS":
    st.title("Evaluation Complete! 📊")
    st.balloons()
    
    results_df = pd.DataFrame(st.session_state.responses)
    
    if not results_df.empty:
        # Calculate overall accuracy
        overall_acc = results_df['human_correct'].mean() * 100
        unanswered_count = len(results_df[results_df['human_answer'] == 'unanswered'])
        
        # Calculate accuracy by mode
        reasoning_subset = results_df[results_df['eval_mode'] == 'Reasoning']
        quick_subset = results_df[results_df['eval_mode'] == 'Quick']
        
        acc_reasoning = reasoning_subset['human_correct'].mean() * 100 if not reasoning_subset.empty else 0
        acc_quick = quick_subset['human_correct'].mean() * 100 if not quick_subset.empty else 0
        
        st.markdown("### Your Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Score", f"{overall_acc:.1f}%")
        col2.metric("Reasoning Score", f"{acc_reasoning:.1f}%")
        col3.metric("Quick Score", f"{acc_quick:.1f}%")
        col4.metric("Timeouts", unanswered_count)
        
        st.markdown("---")
        
        st.markdown("### Save Your Score")
        user_initials = st.text_input("Name/Initials:", placeholder="e.g., John Smith")
        
        # We need a boolean to track if data was successfully submitted
        if 'data_saved' not in st.session_state:
            st.session_state.data_saved = False
            
        if user_initials and not st.session_state.data_saved:
            if st.button("Submit Results to Database", use_container_width=True):
                with st.spinner("Saving to database..."):
                    try:
                        # 1. Initialize Supabase Connection
                        conn = st.connection("supabase", type=SupabaseConnection)
                        
                        # 2. Format the session state list of dicts for the database insertion
                        db_payload = []
                        for response in st.session_state.responses:
                            db_payload.append({
                                "evaluator_name": user_initials,
                                "sample_id": response['id'],
                                "human_answer": response['human_answer'],
                                "human_correct": response['human_correct'],
                                "machine_difficulty": response['machine_difficulty'],
                                "eval_mode": response['eval_mode'],
                                "batch_num": response['batch_num']
                            })
                            
                        # 3. Perform the bulk insert
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