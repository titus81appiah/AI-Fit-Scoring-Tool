import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import StringIO

# Document processing imports
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
        USE_PDFPLUMBER = True
    except ImportError:
        PDF_AVAILABLE = False
        USE_PDFPLUMBER = False

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
if 'reasoning' not in st.session_state:
    st.session_state.reasoning = {}

# Define criteria
criteria = {
    "Data Requirements": {"description": "Availability and quality of data needed for AI implementation", "weight": 25},
    "Problem Complexity": {"description": "How well-suited the problem is for AI/ML solutions", "weight": 20},
    "Business Impact": {"description": "Potential value and ROI of implementing AI solution", "weight": 20},
    "Technical Feasibility": {"description": "Technical constraints and implementation challenges", "weight": 20},
    "Timeline Alignment": {"description": "Compatibility with project timelines and expectations", "weight": 15}
}

def extract_text_from_file(uploaded_file):
    """Extract text from various file formats"""
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    try:
        if file_extension in ['txt', 'md']:
            text = str(uploaded_file.read(), "utf-8")
            return True, text, ""
            
        elif file_extension == 'docx':
            if not DOCX_AVAILABLE:
                return False, "", "python-docx library not available."
            
            doc = Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return True, text, ""
            
        elif file_extension == 'pdf':
            if not PDF_AVAILABLE:
                return False, "", "PDF processing library not available."
            
            text = ""
            if 'USE_PDFPLUMBER' in globals() and USE_PDFPLUMBER:
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages[:5]:  # Limit to first 5 pages
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page_num in range(min(5, pdf_document.page_count)):
                    page = pdf_document[page_num]
                    text += page.get_text() + "\n"
                pdf_document.close()
            
            return True, text, ""
            
        else:
            return False, "", f"Unsupported file format: {file_extension}"
            
    except Exception as e:
        return False, "", f"Error processing file: {str(e)}"

def simple_keyword_analysis(text, criteria_name):
    """Simple keyword-based scoring"""
    if not text:
        return 3, "No text provided"
    
    text_lower = text.lower()
    
    # Define keyword patterns for each criteria
    keyword_patterns = {
        "Data Requirements": {
            "positive": ["data", "dataset", "database", "information", "analytics", "machine learning", "training"],
            "negative": ["no data", "limited data", "poor quality"]
        },
        "Problem Complexity": {
            "positive": ["complex", "advanced", "intelligent", "automated", "optimization", "AI", "machine learning"],
            "negative": ["simple", "basic", "manual", "straightforward"]
        },
        "Business Impact": {
            "positive": ["revenue", "profit", "efficiency", "cost savings", "competitive advantage", "ROI", "value"],
            "negative": ["limited impact", "marginal", "small scale"]
        },
        "Technical Feasibility": {
            "positive": ["feasible", "technology", "platform", "API", "integration", "scalable", "proven"],
            "negative": ["challenging", "difficult", "impossible", "constraints"]
        },
        "Timeline Alignment": {
            "positive": ["timeline", "schedule", "deadline", "priority", "time", "realistic", "phased"],
            "negative": ["urgent", "rushed", "unrealistic", "tight deadline"]
        }
    }
    
    if criteria_name not in keyword_patterns:
        return 3, "No keyword pattern defined"
    
    pattern = keyword_patterns[criteria_name]
    
    # Count positive and negative keywords
    positive_matches = sum(1 for kw in pattern["positive"] if kw in text_lower)
    negative_matches = sum(1 for kw in pattern["negative"] if kw in text_lower)
    
    # Calculate score (1-5 scale)
    base_score = 3  # Start with neutral
    base_score += min(positive_matches * 0.3, 2)  # Add up to 2 points for positive
    base_score -= min(negative_matches * 0.5, 1.5)  # Subtract up to 1.5 for negative
    
    final_score = max(1, min(5, round(base_score)))
    
    # Generate reasoning
    reasoning_parts = []
    if positive_matches > 0:
        reasoning_parts.append(f"✓ Found {positive_matches} positive indicators")
    if negative_matches > 0:
        reasoning_parts.append(f"⚠ Found {negative_matches} concerns")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Limited relevant keywords found"
    
    return final_score, reasoning

def get_project_type(text):
    """Simple project type detection"""
    if not text:
        return "Unknown"
    
    text_lower = text.lower()
    project_indicators = {
        'Web Application': ['web', 'website', 'webapp', 'frontend', 'backend', 'api'],
        'Mobile Application': ['mobile', 'app', 'ios', 'android', 'smartphone'],
        'Data Analysis': ['data', 'analytics', 'dashboard', 'reporting', 'visualization'],
        'Machine Learning': ['ml', 'ai', 'prediction', 'classification', 'model'],
        'Automation': ['automate', 'workflow', 'process', 'efficiency', 'streamline']
    }
    
    scores = {}
    for project_type, indicators in project_indicators.items():
        score = sum(1 for indicator in indicators if indicator in text_lower)
        scores[project_type] = score
    
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "General Software"

# Header
st.title("🎯 AI Fit Scoring Dashboard")
st.markdown("**Phase 2: With Document Processing** 📄")
st.write("Testing document upload and text extraction capabilities")

# Sidebar for input method
st.sidebar.title("📋 Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["📄 Upload Document", "✍️ Manual Scoring"],
    key="input_method"
)

# Document Upload Section
if input_method == "📄 Upload Document":
    st.subheader("📤 Document Upload & Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload your project document",
        type=['pdf', 'docx', 'txt', 'md'],
        help="Supported formats: PDF, DOCX, TXT, MD",
        key="document_uploader"
    )
    
    if uploaded_file is not None:
        # File info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"✅ {uploaded_file.name}")
        with col2:
            st.info(f"📊 {uploaded_file.size / 1024:.1f} KB")
        
        # Extract text
        with st.spinner("🔄 Processing document..."):
            success, extracted_text, error_message = extract_text_from_file(uploaded_file)
        
        if success and extracted_text.strip():
            # Document statistics
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)
            project_type = get_project_type(extracted_text)
            
            with col3:
                st.metric("📝 Words", f"{word_count:,}")
            
            col4, col5 = st.columns(2)
            with col4:
                st.metric("🔤 Characters", f"{char_count:,}")
            with col5:
                st.metric("🎯 Detected Type", project_type)
            
            # Document preview
            with st.expander("📖 Document Preview"):
                preview_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                st.text_area("Content Preview:", preview_text, height=200, disabled=True, key="preview")
            
            # Auto-scoring buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 Auto-Score Document", type="primary", key="auto_score"):
                    with st.spinner("🔍 Analyzing document..."):
                        for criteria_name in criteria.keys():
                            score, reasoning = simple_keyword_analysis(extracted_text, criteria_name)
                            st.session_state.scores[criteria_name] = score
                            st.session_state.reasoning[criteria_name] = reasoning
                    
                    st.success("🎯 Auto-scoring completed!")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset Scores", key="reset"):
                    st.session_state.scores = {}
                    st.session_state.reasoning = {}
                    st.rerun()
                    
        else:
            st.error(f"❌ Error: {error_message}")

# Manual Scoring Section
elif input_method == "✍️ Manual Scoring":
    st.subheader("✍️ Manual Project Scoring")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**📊 Scoring Interface**")
        
        for criteria_name, criteria_info in criteria.items():
            # Get current values
            current_score = st.session_state.scores.get(criteria_name, 3)
            current_reasoning = st.session_state.reasoning.get(criteria_name, "")
            
            score = st.slider(
                f"{criteria_name} ({criteria_info['weight']}%)",
                1, 5, current_score,
                help=criteria_info["description"],
                key=f"score_{criteria_name}"
            )
            
            reasoning = st.text_area(
                f"Reasoning for {criteria_name}",
                current_reasoning,
                height=60,
                key=f"reasoning_{criteria_name}",
                placeholder="Optional: Explain your scoring rationale..."
            )
            
            # Update session state
            st.session_state.scores[criteria_name] = score
            st.session_state.reasoning[criteria_name] = reasoning
    
    with col2:
        st.write("**📈 Live Results**")
        # Results will show here (handled in the results section below)

# Results Section (shown for both methods)
if st.session_state.scores:
    st.markdown("---")
    st.subheader("📊 AI Fit Assessment Results")
    
    # Calculate overall score
    total_weighted_score = 0
    for criteria_name, criteria_info in criteria.items():
        if criteria_name in st.session_state.scores:
            score = st.session_state.scores[criteria_name]
            percentage = (score - 1) / 4 * 100
            weight = criteria_info["weight"]
            total_weighted_score += percentage * (weight / 100)
    
    # Display main score with enhanced styling
    score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
    with score_col2:
        if total_weighted_score >= 70:
            st.success(f"## 🟢 {total_weighted_score:.0f}% - Excellent AI Fit")
            recommendation = "✅ **Highly Recommended** - This project shows strong potential for AI implementation."
        elif total_weighted_score >= 60:
            st.success(f"## 🔵 {total_weighted_score:.0f}% - Good AI Fit") 
            recommendation = "✅ **Recommended** - This project has good AI potential with proper planning."
        elif total_weighted_score >= 40:
            st.warning(f"## 🟡 {total_weighted_score:.0f}% - Moderate AI Fit")
            recommendation = "⚠️ **Proceed with Caution** - Evaluate risks and benefits carefully."
        else:
            st.error(f"## 🔴 {total_weighted_score:.0f}% - Low AI Fit")
            recommendation = "❌ **Not Recommended** - Consider alternative approaches."
        
        st.markdown(recommendation)
    
    # Detailed scoring cards
    st.subheader("📋 Detailed Score Breakdown")
    
    for criteria_name, criteria_info in criteria.items():
        if criteria_name in st.session_state.scores:
            score = st.session_state.scores[criteria_name]
            percentage = (score - 1) / 4 * 100
            reasoning = st.session_state.reasoning.get(criteria_name, "No reasoning provided")
            
            # Color coding
            if percentage >= 75:
                color = "🟢"
                status = "Excellent"
            elif percentage >= 50:
                color = "🔵" 
                status = "Good"
            elif percentage >= 25:
                color = "🟡"
                status = "Average"
            else:
                color = "🔴"
                status = "Poor"
            
            with st.container():
                st.markdown(f"""
                **{color} {criteria_name}** - {percentage:.0f}% ({status})  
                *Weight: {criteria_info['weight']}% | Raw Score: {score}/5*  
                📝 {reasoning}
                """)
    
    # Visualization Section
    st.markdown("---")
    st.subheader("📊 Visual Analysis")
    
    # Prepare data
    criteria_names = list(st.session_state.scores.keys())
    scores = list(st.session_state.scores.values())
    percentages = [(score - 1) / 4 * 100 for score in scores]
    
    # Create charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#d32f2f' if p < 40 else '#f57c00' if p < 60 else '#388e3c' if p < 75 else '#1976d2' for p in percentages]
        
        bars = ax.bar(range(len(criteria_names)), percentages, color=colors, alpha=0.8)
        ax.set_xlabel('Criteria')
        ax.set_ylabel('Score (%)')
        ax.set_title('AI Fit Score Breakdown', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(criteria_names)))
        ax.set_xticklabels([name.replace(' ', '\n') for name in criteria_names], fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add score labels
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{percentage:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with chart_col2:
        # Radar Chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(criteria_names), endpoint=False).tolist()
        percentages_plot = percentages + [percentages[0]]
        angles += angles[:1]
        
        ax.plot(angles, percentages_plot, 'o-', linewidth=2, color='#1976d2')
        ax.fill(angles, percentages_plot, alpha=0.25, color='#1976d2')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([name.replace(' ', '\n') for name in criteria_names])
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.set_title("AI Fit Profile", size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Export functionality
    st.subheader("📤 Export Results")
    
    if st.button("📥 Download Results (CSV)", key="download"):
        # Create detailed report
        report_data = []
        for criteria_name in criteria.keys():
            if criteria_name in st.session_state.scores:
                score = st.session_state.scores[criteria_name]
                percentage = (score - 1) / 4 * 100
                report_data.append({
                    'Criteria': criteria_name,
                    'Score (1-5)': score,
                    'Percentage': f"{percentage:.0f}%",
                    'Weight': f"{criteria[criteria_name]['weight']}%",
                    'Weighted Score': f"{percentage * criteria[criteria_name]['weight'] / 100:.1f}",
                    'Reasoning': st.session_state.reasoning.get(criteria_name, ""),
                    'Description': criteria[criteria_name]['description']
                })
        
        if report_data:
            df = pd.DataFrame(report_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"ai_fit_assessment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# Status indicators
st.markdown("---")
st.subheader("🔧 System Status")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"📄 **PDF Processing:** {'✅ Available' if PDF_AVAILABLE else '❌ Not Available'}")
with col2:
    st.write(f"📄 **DOCX Processing:** {'✅ Available' if DOCX_AVAILABLE else '❌ Not Available'}")
with col3:
    st.write(f"📊 **Visualization:** ✅ Available")

st.markdown("**✅ Phase 2 Complete:** Document processing capabilities added!")
st.markdown("**🎯 Next:** If this works, we'll add basic NLP analysis (scikit-learn + textblob).")