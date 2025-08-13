import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking charts
plt.style.use('default')
sns.set_palette("husl")

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
st.markdown("**Phase 1: With Visualization** 📊")
st.write("Testing matplotlib and seaborn integration")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Scoring Interface")
    
    for criteria_name, criteria_info in criteria.items():
        score = st.slider(
            f"{criteria_name} ({criteria_info['weight']}%)",
            1, 5, 3,
            help=criteria_info["description"],
            key=f"score_{criteria_name}"
        )
        st.session_state.scores[criteria_name] = score

with col2:
    st.subheader("📈 Results")
    
    # Calculate overall score
    if st.session_state.scores:
        total_weighted_score = 0
        for criteria_name, criteria_info in criteria.items():
            if criteria_name in st.session_state.scores:
                score = st.session_state.scores[criteria_name]
                percentage = (score - 1) / 4 * 100
                weight = criteria_info["weight"]
                total_weighted_score += percentage * (weight / 100)
        
        # Display main score with color coding
        if total_weighted_score >= 70:
            st.success(f"🟢 **{total_weighted_score:.0f}%** - Excellent AI Fit")
        elif total_weighted_score >= 60:
            st.success(f"🔵 **{total_weighted_score:.0f}%** - Good AI Fit")
        elif total_weighted_score >= 40:
            st.warning(f"🟡 **{total_weighted_score:.0f}%** - Moderate AI Fit")
        else:
            st.error(f"🔴 **{total_weighted_score:.0f}%** - Low AI Fit")
        
        # Show individual scores
        st.write("**Score Breakdown:**")
        for criteria_name in criteria.keys():
            if criteria_name in st.session_state.scores:
                score = st.session_state.scores[criteria_name]
                percentage = (score - 1) / 4 * 100
                st.write(f"• {criteria_name}: {percentage:.0f}%")

# Visualization Section
if st.session_state.scores:
    st.markdown("---")
    st.subheader("📊 Visual Analysis")
    
    # Prepare data for visualization
    criteria_names = list(st.session_state.scores.keys())
    scores = list(st.session_state.scores.values())
    percentages = [(score - 1) / 4 * 100 for score in scores]
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color bars based on score
        colors = ['#d32f2f' if p < 40 else '#f57c00' if p < 60 else '#388e3c' if p < 80 else '#1976d2' for p in percentages]
        
        bars = ax.bar(range(len(criteria_names)), percentages, color=colors, alpha=0.8)
        ax.set_xlabel('Criteria')
        ax.set_ylabel('Score (%)')
        ax.set_title('AI Fit Score Breakdown', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(criteria_names)))
        ax.set_xticklabels([name.replace(' ', '\n') for name in criteria_names], fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add score labels on bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{percentage:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with chart_col2:
        # Radar Chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Complete the circle
        angles = np.linspace(0, 2 * np.pi, len(criteria_names), endpoint=False).tolist()
        percentages_plot = percentages + [percentages[0]]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, percentages_plot, 'o-', linewidth=2, color='#1976d2')
        ax.fill(angles, percentages_plot, alpha=0.25, color='#1976d2')
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([name.replace(' ', '\n') for name in criteria_names])
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.set_title("AI Fit Profile", size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Highest Score", f"{max(percentages):.0f}%")
    with col2:
        st.metric("Lowest Score", f"{min(percentages):.0f}%")
    with col3:
        st.metric("Average Score", f"{np.mean(percentages):.0f}%")
    with col4:
        st.metric("Score Range", f"{max(percentages) - min(percentages):.0f}%")
    
    # Data Table
    with st.expander("📋 Detailed Breakdown"):
        results_df = pd.DataFrame({
            'Criteria': criteria_names,
            'Raw Score (1-5)': scores,
            'Percentage': [f"{p:.0f}%" for p in percentages],
            'Weight': [f"{criteria[name]['weight']}%" for name in criteria_names],
            'Weighted Score': [f"{p * criteria[name]['weight'] / 100:.1f}" for p, name in zip(percentages, criteria_names)],
            'Status': ['Excellent' if p >= 80 else 'Good' if p >= 60 else 'Average' if p >= 40 else 'Poor' for p in percentages]
        })
        st.dataframe(results_df, use_container_width=True)

st.markdown("---")
st.markdown("**✅ Phase 1 Complete:** Basic visualization with matplotlib and seaborn working!")
st.markdown("**🎯 Next:** If this works, we'll add document upload functionality.")