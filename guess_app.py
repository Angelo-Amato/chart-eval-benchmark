import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURATION ---
DATA_PATH = "human_eval_set.csv"
IMAGE_DIR = "data/raw"
RENDER_BUFFER = 2.0  # Seconds to wait for the image to load before strictly counting down

st.set_page_config(page_title="Chart Comparison Human Eval", layout="centered")

# --- INITIALIZE SESSION STATE ---
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'quiz_complete' not in st.session_state:
    st.session_state.quiz_complete = False
if 'eval_started' not in st.session_state:
    st.session_state.eval_started = False
if 'mode' not in st.session_state:
    st.session_state.mode = "Reasoning"
if 'time_limit' not in st.session_state:
    st.session_state.time_limit = 10
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Cache the image reading so we don't hit the hard drive every 1 second during countdown
@st.cache_data
def load_image_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

df = load_data()
total_questions = len(df)

def record_answer(answer):
    current_row = df.iloc[st.session_state.current_idx]
    
    # Check if human is correct
    if answer == "unanswered":
        is_correct = False
    else:
        human_pred_diff = (answer == "Different")
        is_correct = (human_pred_diff == current_row['actual_diff'])
    
    st.session_state.responses.append({
        'id': current_row['id'],
        'human_answer': answer,
        'human_correct': is_correct,
        'machine_difficulty': current_row['machine_difficulty']
    })
    
    if st.session_state.current_idx < total_questions - 1:
        st.session_state.current_idx += 1
        # Add a render buffer to the start time to account for network/image loading latency
        st.session_state.start_time = time.time() + RENDER_BUFFER  
    else:
        st.session_state.quiz_complete = True

# --- SETUP SCREEN ---
if not st.session_state.eval_started:
    st.title("Human Evaluation Setup")
    st.markdown("Select your evaluation mode:")
    
    st.session_state.mode = st.radio("Mode", ["Reasoning", "Quick"], horizontal=True)
    
    if st.session_state.mode == "Quick":
        st.session_state.time_limit = st.number_input("Time limit per sample (seconds):", min_value=3, max_value=60, value=10)
        st.info(f"Each chart will be displayed for a maximum of {st.session_state.time_limit} seconds. If time expires, the sample is marked as 'unanswered'.")
    else:
        st.info("Take as much time as you need to analyze each chart.")
        
    if st.button("Start Evaluation", use_container_width=True):
        st.session_state.eval_started = True
        # Add a render buffer to the start time to account for initial image loading
        st.session_state.start_time = time.time() + RENDER_BUFFER
        st.rerun()

# --- QUIZ INTERFACE ---
elif not st.session_state.quiz_complete:
    
    # Trigger background auto-refresh every 1000ms (1 sec) ONLY in Quick mode
    if st.session_state.mode == "Quick":
        st_autorefresh(interval=1000, key=f"timer_{st.session_state.current_idx}")
        
    st.title("Human Evaluation: Chart Comparison")
    st.progress((st.session_state.current_idx) / total_questions)
    st.write(f"**Question {st.session_state.current_idx + 1} of {total_questions}** | Mode: **{st.session_state.mode}**")
    
    # Timer Logic
    if st.session_state.mode == "Quick":
        elapsed = time.time() - st.session_state.start_time
        # Prevent remaining time from showing higher than the limit during the buffer phase
        remaining = min(st.session_state.time_limit, int(st.session_state.time_limit - elapsed))
        
        if remaining <= 0:
            record_answer("unanswered")
            st.rerun()
        else:
            st.warning(f"⏱️ Time Remaining: **{remaining}s**")

    current_row = df.iloc[st.session_state.current_idx]
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
        st.button("They are the SAME", key=f"same_{st.session_state.current_idx}", on_click=record_answer, args=("Same",), use_container_width=True)
    with col2:
        st.button("They are DIFFERENT", key=f"diff_{st.session_state.current_idx}", on_click=record_answer, args=("Different",), use_container_width=True)

# --- RESULTS DASHBOARD ---
else:
    st.title("Evaluation Complete! 🎉")
    st.balloons()
    
    results_df = pd.DataFrame(st.session_state.responses)
    
    # Calculate Correlations
    hard_subset = results_df[results_df['machine_difficulty'] == 'Hard']
    easy_subset = results_df[results_df['machine_difficulty'] == 'Easy']
    
    human_acc_hard = hard_subset['human_correct'].mean() * 100 if len(hard_subset) > 0 else 0
    human_acc_easy = easy_subset['human_correct'].mean() * 100 if len(easy_subset) > 0 else 0
    overall_acc = results_df['human_correct'].mean() * 100
    unanswered_count = len(results_df[results_df['human_answer'] == 'unanswered'])
    
    st.markdown("### Human vs. Machine Correlation")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Score", f"{overall_acc:.1f}%")
    col2.metric("Score on 'Easy'", f"{human_acc_easy:.1f}%")
    col3.metric("Score on 'Hard'", f"{human_acc_hard:.1f}%")
    col4.metric("Unanswered", unanswered_count)
    
    st.markdown("---")
    if human_acc_hard < human_acc_easy:
        st.success("Correlation Found: Humans also struggled with the charts the AI found difficult.")
    else:
        st.warning("Divergence Found: The AI's blindspots do not strongly correlate with human visual reasoning blindspots.")
        
    st.markdown("### Save Your Score")
    st.write("Enter your name or initials to sign your results:")
    
    user_initials = st.text_input("Name/Initials:", placeholder="e.g., John Smith")
    
    if user_initials:
        current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_df["evaluator_name"] = user_initials
        results_df["evaluation_date"] = current_date_str
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        file_date = datetime.now().strftime("%Y%m%d")
        
        # Format filename to be safe (replace spaces with underscores)
        safe_name = user_initials.replace(" ", "_")
        download_filename = f'human_eval_results_{safe_name}_{file_date}.csv'
        
        st.download_button(
            label="Download Human Eval Results as CSV",
            data=csv,
            file_name=download_filename,
            mime='text/csv',
        )
    else:
        st.info("Please enter your name or initials above to unlock the download button.")
