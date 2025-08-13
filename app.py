import streamlit as st
import pandas as pd

# Streamlit App Configuration
st.set_page_config(
    page_title="AI Fit Scoring Dashboard",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'scores' not in st.session_state:
    st.session_state.scores = {}

# Define criteria
criteria = {
    "Data Requirements": {"description": "Availability and quality of data needed for AI implementation", "weight": 25},
    "Problem Complexity": {"description": "How well-suited the problem is for AI/ML solutions", "weight": 20},
    "Business Impact": {"description": "Potential value and ROI of implementing AI solution", "weight": 20},
    "Technical Feasibility": {"description": "Technical constraints and implementation challenges", "weight": 20},
    "Timeline Alignment": {"description": "Compatibility with project timelines and expectations", "weight": 15}
}

# Header
st.title("🎯 AI Fit Scoring Dashboard")
st.write("Minimal working version - testing deployment")

# Basic scoring interface
st.subheader("Basic Scoring")

for criteria_name, criteria_info in criteria.items():
    score = st.slider(
        f"{criteria_name} ({criteria_info['weight']}%)",
        1, 5, 3,
        help=criteria_info["description"],
        key=f"score_{criteria_name}"
    )
    st.session_state.scores[criteria_name] = score

# Calculate overall score
if st.session_state.scores:
    total_weighted_score = 0
    for criteria_name, criteria_info in criteria.items():
        if criteria_name in st.session_state.scores:
            score = st.session_state.scores[criteria_name]
            percentage = (score - 1) / 4 * 100
            weight = criteria_info["weight"]
            total_weighted_score += percentage * (weight / 100)
    
    st.metric("Overall AI Fit Score", f"{total_weighted_score:.0f}%")
    
    if total_weighted_score >= 60:
        st.success("✅ Good AI Fit - Recommended")
    elif total_weighted_score >= 40:
        st.warning("⚠️ Moderate AI Fit - Proceed with caution")
    else:
        st.error("❌ Low AI Fit - Not recommended")

st.write("---")
st.write("✅ If you see this, the basic app is working! We can now add features back gradually.")