import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re

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

def extract_text_from_file(uploaded_file):
    """
    Extract text from various file formats
    Returns: (success: bool, text: str, error_message: str)
    """
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    try:
        if file_extension == 'txt':
            # Handle text files
            text = str(uploaded_file.read(), "utf-8")
            return True, text, ""
            
        elif file_extension == 'md':
            # Handle markdown files
            text = str(uploaded_file.read(), "utf-8")
            return True, text, ""
            
        elif file_extension == 'docx':
            if not DOCX_AVAILABLE:
                return False, "", "python-docx library not installed. Please install it to process Word documents."
            
            # Handle Word documents
            doc = Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return True, text, ""
            
        elif file_extension == 'pdf':
            if not PDF_AVAILABLE:
                return False, "", "PDF processing library not installed. Please install PyMuPDF or pdfplumber."
            
            # Handle PDF files
            text = ""
            if 'USE_PDFPLUMBER' in globals() and USE_PDFPLUMBER:
                # Using pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                # Using PyMuPDF
                pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text += page.get_text() + "\n"
                pdf_document.close()
            
            return True, text, ""
            
        else:
            return False, "", f"Unsupported file format: {file_extension}"
            
    except Exception as e:
        return False, "", f"Error processing file: {str(e)}"

def auto_score_criteria(text, criteria_name, keywords, context_keywords=None):
    """
    Automatically score a criteria based on text analysis
    Returns: (score: int, reasoning: str)
    """
    text_lower = text.lower()
    
    # Count keyword matches
    keyword_matches = []
    for keyword in keywords:
        if keyword.lower() in text_lower:
            keyword_matches.append(keyword)
    
    # Count context keyword matches if provided
    context_matches = []
    if context_keywords:
        for keyword in context_keywords:
            if keyword.lower() in text_lower:
                context_matches.append(keyword)
    
    # Simple scoring logic
    keyword_ratio = len(keyword_matches) / len(keywords) if keywords else 0
    context_ratio = len(context_matches) / len(context_keywords) if context_keywords else 0
    
    # Combined score calculation
    if keyword_ratio >= 0.7:  # High keyword presence
        base_score = 4
    elif keyword_ratio >= 0.4:
        base_score = 3
    elif keyword_ratio >= 0.2:
        base_score = 2
    else:
        base_score = 1
    
    # Adjust based on context
    if context_keywords and context_ratio > 0.5:
        base_score = min(5, base_score + 1)
    
    # Generate reasoning
    reasoning = f"Found {len(keyword_matches)} relevant keywords: {', '.join(keyword_matches[:3])}{'...' if len(keyword_matches) > 3 else ''}"
    if context_matches:
        reasoning += f". Context indicators: {', '.join(context_matches[:2])}{'...' if len(context_matches) > 2 else ''}"
    
    return base_score, reasoning

def get_auto_scoring_keywords():
    """Define keywords for automatic scoring of each criteria"""
    return {
        "Data Requirements": {
            "keywords": ["data", "dataset", "training data", "machine learning", "algorithm", "model", "prediction", "classification", "regression", "neural network"],
            "context": ["big data", "data quality", "data preprocessing", "feature engineering", "data pipeline"]
        },
        "Problem Complexity": {
            "keywords": ["complex", "sophisticated", "advanced", "optimization", "automation", "intelligent", "adaptive", "scalable"],
            "context": ["simple task", "basic requirement", "straightforward", "manual process", "rule-based"]
        },
        "Business Impact": {
            "keywords": ["revenue", "cost", "efficiency", "productivity", "competitive advantage", "ROI", "business value", "strategic"],
            "context": ["million", "thousand", "percent", "improvement", "reduction", "increase", "growth"]
        },
        "Technical Feasibility": {
            "keywords": ["feasible", "possible", "achievable", "technology", "platform", "infrastructure", "API", "integration"],
            "context": ["challenging", "difficult", "impossible", "limited resources", "constraints"]
        },
        "Timeline Alignment": {
            "keywords": ["timeline", "deadline", "schedule", "urgent", "priority", "quick", "fast", "immediate"],
            "context": ["weeks", "months", "days", "quarter", "year", "ASAP", "by"]
        }
    }

# Streamlit App Configuration
st.set_page_config(
    page_title="AI Fit Scoring Tool",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Fit Scoring Tool")
st.markdown("Evaluate project proposals for AI implementation suitability")

# Sidebar for method selection
st.sidebar.title("📋 Input Method")
input_method = st.sidebar.radio(
    "Choose how to provide project information:",
    ["📄 Upload Document", "✍️ Manual Entry"]
)

# Initialize session state
if 'scores' not in st.session_state:
    st.session_state.scores = {}
if 'reasoning' not in st.session_state:
    st.session_state.reasoning = {}

# Define scoring criteria
criteria = {
    "Data Requirements": {
        "description": "Availability and quality of data needed for AI implementation",
        "weight": 0.25
    },
    "Problem Complexity": {
        "description": "How well-suited the problem is for AI/ML solutions",
        "weight": 0.20
    },
    "Business Impact": {
        "description": "Potential value and ROI of implementing AI solution",
        "weight": 0.20
    },
    "Technical Feasibility": {
        "description": "Technical constraints and implementation challenges",
        "weight": 0.20
    },
    "Timeline Alignment": {
        "description": "Compatibility with project timelines and expectations",
        "weight": 0.15
    }
}

# Document Upload Section
if input_method == "📄 Upload Document":
    st.header("📤 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your project document (PDF, DOCX, TXT, MD)",
        type=['pdf', 'docx', 'txt', 'md'],
        help="Upload a document containing project requirements, scope, or proposal"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Extract text from file
        with st.spinner("🔄 Processing document..."):
            success, extracted_text, error_message = extract_text_from_file(uploaded_file)
        
        if success:
            # Show extracted text preview
            with st.expander("📖 Document Preview"):
                st.text_area("Extracted Text (first 500 characters):", 
                           extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                           height=150)
            
            # Auto-scoring option
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 Auto Score Document", type="primary"):
                    keywords_config = get_auto_scoring_keywords()
                    
                    with st.spinner("🔍 Analyzing document..."):
                        for criteria_name in criteria.keys():
                            if criteria_name in keywords_config:
                                config = keywords_config[criteria_name]
                                score, reasoning = auto_score_criteria(
                                    extracted_text, 
                                    criteria_name,
                                    config["keywords"],
                                    config.get("context")
                                )
                                st.session_state.scores[criteria_name] = score
                                st.session_state.reasoning[criteria_name] = reasoning
                    
                    st.success("🎯 Auto-scoring completed!")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset Scores"):
                    st.session_state.scores = {}
                    st.session_state.reasoning = {}
                    st.rerun()
                    
        else:
            st.error(f"❌ Error processing file: {error_message}")
            if not DOCX_AVAILABLE and uploaded_file.name.endswith('.docx'):
                st.info("💡 To process Word documents, install: `pip install python-docx`")
            elif not PDF_AVAILABLE and uploaded_file.name.endswith('.pdf'):
                st.info("💡 To process PDFs, install: `pip install PyMuPDF` or `pip install pdfplumber`")

# Manual Entry Section
elif input_method == "✍️ Manual Entry":
    st.header("✍️ Manual Project Entry")
    project_description = st.text_area(
        "Project Description",
        height=150,
        placeholder="Describe your project, requirements, and objectives..."
    )

# Scoring Interface
st.header("📊 AI Fit Scoring")

col1, col2 = st.columns([2, 1])

with col1:
    for criteria_name, criteria_info in criteria.items():
        st.subheader(f"{criteria_name} ({criteria_info['weight']*100:.0f}%)")
        st.caption(criteria_info["description"])
        
        # Get current score from session state
        current_score = st.session_state.scores.get(criteria_name, 3)
        current_reasoning = st.session_state.reasoning.get(criteria_name, "")
        
        # Score slider
        score = st.select_slider(
            f"Score for {criteria_name}",
            options=[1, 2, 3, 4, 5],
            value=current_score,
            format_func=lambda x: f"{x} - {'Poor' if x==1 else 'Below Average' if x==2 else 'Average' if x==3 else 'Good' if x==4 else 'Excellent'}",
            key=f"score_{criteria_name}"
        )
        
        # Reasoning text area
        reasoning = st.text_area(
            f"Reasoning for {criteria_name}",
            value=current_reasoning,
            height=100,
            key=f"reasoning_{criteria_name}",
            placeholder="Explain your scoring rationale..."
        )
        
        # Update session state
        st.session_state.scores[criteria_name] = score
        st.session_state.reasoning[criteria_name] = reasoning
        
        st.divider()

# Results Panel
with col2:
    st.subheader("📈 Results Summary")
    
    if st.session_state.scores:
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        
        for criteria_name, criteria_info in criteria.items():
            if criteria_name in st.session_state.scores:
                score = st.session_state.scores[criteria_name]
                weight = criteria_info["weight"]
                total_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_average = total_score / total_weight
            
            # Display overall score
            st.metric("Overall AI Fit Score", f"{weighted_average:.2f}/5.00")
            
            # Recommendation
            if weighted_average >= 4.0:
                st.success("🟢 **High AI Fit** - Strongly recommended")
            elif weighted_average >= 3.0:
                st.warning("🟡 **Medium AI Fit** - Consider with caution")
            else:
                st.error("🔴 **Low AI Fit** - Not recommended")
            
            # Score breakdown
            st.subheader("Score Breakdown")
            for criteria_name in criteria.keys():
                if criteria_name in st.session_state.scores:
                    score = st.session_state.scores[criteria_name]
                    st.write(f"**{criteria_name}:** {score}/5")
            
            # Visualization
            if len(st.session_state.scores) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                criteria_names = list(st.session_state.scores.keys())
                scores = list(st.session_state.scores.values())
                
                bars = ax.barh(criteria_names, scores, color='skyblue')
                ax.set_xlim(0, 5)
                ax.set_xlabel('Score')
                ax.set_title('AI Fit Scoring Results')
                
                # Add score labels on bars
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                           str(score), va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Export results
                st.subheader("📤 Export Results")
                
                # Create summary report
                report_data = []
                for criteria_name in criteria.keys():
                    if criteria_name in st.session_state.scores:
                        report_data.append({
                            'Criteria': criteria_name,
                            'Score': st.session_state.scores[criteria_name],
                            'Weight': criteria[criteria_name]['weight'],
                            'Weighted Score': st.session_state.scores[criteria_name] * criteria[criteria_name]['weight'],
                            'Reasoning': st.session_state.reasoning.get(criteria_name, "")
                        })
                
                if report_data:
                    df = pd.DataFrame(report_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"ai_fit_scoring_results.csv",
                        mime="text/csv"
                    )

# Footer
st.markdown("---")
st.markdown("**AI Fit Scoring Tool** - Evaluate projects for AI implementation potential")