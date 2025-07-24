from draw_step import draw_step
from threshold_step import threshold_step
from process_step import process_step
import streamlit as st

# ================== Main App ==================
def main():
    st.set_page_config(layout="wide")
    st.title("# Density Segmentation GUI")
    
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "draw"
    
    if st.session_state["current_step"] == "draw":
        draw_step()
    elif st.session_state["current_step"] == "threshold":
        threshold_step()
    elif st.session_state["current_step"] == "process":
        process_step()

if __name__ == "__main__":
    main()