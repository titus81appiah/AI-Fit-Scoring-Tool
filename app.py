import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re
import numpy as np
from collections import Counter
import warnings
import time
warnings.filterwarnings('ignore')

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

# NLP imports
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        st.error("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    # Initialize sentiment analysis pipeline
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", 
                                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                    return_all_scores=True)
        SENTIMENT_AVAILABLE = True
    except:
        SENTIMENT_AVAILABLE = False
        sentiment_pipeline = None
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    SENTIMENT_AVAILABLE = False
    sentiment_pipeline = None

# Custom CSS for improved dashboard styling
st.markdown("""
<style>
    .main-metric {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 3px solid #667eea;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        border: 1px solid #f0f0f0;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.12);
    }
    
    .score-excellent { border-left-color: #28a745 !important; }
    .score-good { border-left-color: #17a2b8 !important; }
    .score-average { border-left-color: #ffc107 !important; }
    .score-poor { border-left-color: #dc3545 !important; }
    
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .dashboard-header h1 {
        color: white !important;
        margin-bottom: 0.5rem;
    }
    
    .dashboard-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem;
    }
    
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0.3rem 0;
    }
    
    .badge-excellent {
        background: rgba(40, 167, 69, 0.1);
        color: #28a745;
        border: 1px solid rgba(40, 167, 69, 0.3);
    }
    
    .badge-good {
        background: rgba(23, 162, 184, 0.1);
        color: #17a2b8;
        border: 1px solid rgba(23, 162, 184, 0.3);
    }
    
    .badge-average {
        background: rgba(255, 193, 7, 0.1);
        color: #e0a800;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    
    .badge-poor {
        background: rgba(220, 53, 69, 0.1);
        color: #dc3545;
        border: 1px solid rgba(220, 53, 69, 0.3);
    }
    
    .overall-score-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        border: 2px solid #dee2e6;
        position: relative;
        overflow: hidden;
    }
    
    .overall-score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #28a745, #17a2b8, #ffc107, #dc3545);
    }
    
    .status-indicator {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-available {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-unavailable {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .processing-time {
        font-size: 0.8rem;
        color: #6c757d;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class NLPProcessor:
    def __init__(self):
        self.nlp = nlp
        self.tfidf = None
        if SKLEARN_AVAILABLE:
            self.tfidf = TfidfVectorizer(
                max_features=500,  # Reduced for faster processing
                stop_words='english',
                ngram_range=(1, 2),  # Reduced range
                min_df=1
            )
        self.cache = {}  # Simple cache for processed results
    
    def preprocess_text(self, text):
        """Optimized text preprocessing"""
        if not text:
            return ""
        
        # Check cache
        text_hash = hash(text)
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = text.lower().strip()
        
        # Cache result
        self.cache[text_hash] = text
        return text
    
    @st.cache_data
    def extract_entities(_self, text):
        """Extract named entities with caching"""
        if not _self.nlp or not text:
            return []
        
        doc = _self.nlp(text[:2000])  # Limit text length for speed
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'TECHNOLOGY', 'MONEY', 'PERCENT']:  # Focus on relevant entities
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_)
                })
        
        return entities[:10]  # Limit results
    
    @st.cache_data
    def extract_keywords(_self, text, top_k=8):
        """Optimized keyword extraction with caching"""
        if not SKLEARN_AVAILABLE or not text:
            return []
        
        try:
            processed_text = _self.preprocess_text(text)
            
            # Fit TF-IDF
            tfidf_matrix = _self.tfidf.fit_transform([processed_text])
            feature_names = _self.tfidf.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = np.argsort(tfidf_scores)[::-1][:top_k]
            keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
            
            return keywords
        except:
            return []
    
    @st.cache_data
    def analyze_sentiment(_self, text):
        """Cached sentiment analysis"""
        if not SENTIMENT_AVAILABLE or not text:
            return None
        
        try:
            # Limit text for faster processing
            text_sample = text[:512]
            results = sentiment_pipeline(text_sample)
            if results and len(results[0]) > 0:
                best_result = max(results[0], key=lambda x: x['score'])
                return {
                    'label': best_result['label'],
                    'confidence': best_result['score']
                }
        except:
            return None
        
        return None
    
    def extract_project_type(self, text):
        """Fast project type detection"""
        text_lower = text.lower()
        
        project_indicators = {
            'Automation': ['automate', 'workflow', 'process', 'efficiency', 'streamline'],
            'Data Analysis': ['data', 'analytics', 'dashboard', 'reporting', 'visualization'],
            'Machine Learning': ['ml', 'ai', 'prediction', 'classification', 'model'],
            'Web Application': ['web', 'website', 'webapp', 'frontend', 'api'],
            'Mobile Application': ['mobile', 'app', 'ios', 'android']
        }
        
        scores = {}
        for project_type, indicators in project_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[project_type] = score
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "General"

def extract_text_from_file(uploaded_file):
    """Optimized file text extraction"""
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    try:
        if file_extension in ['txt', 'md']:
            text = str(uploaded_file.read(), "utf-8")
            return True, text, ""
            
        elif file_extension == 'docx':
            if not DOCX_AVAILABLE:
                return False, "", "python-docx library not installed."
            
            doc = Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return True, text, ""
            
        elif file_extension == 'pdf':
            if not PDF_AVAILABLE:
                return False, "", "PDF processing library not installed."
            
            text = ""
            if 'USE_PDFPLUMBER' in globals() and USE_PDFPLUMBER:
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages[:5]:  # Limit to first 5 pages for speed
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page_num in range(min(5, pdf_document.page_count)):  # Limit pages
                    page = pdf_document[page_num]
                    text += page.get_text() + "\n"
                pdf_document.close()
            
            return True, text, ""
            
        else:
            return False, "", f"Unsupported file format: {file_extension}"
            
    except Exception as e:
        return False, "", f"Error processing file: {str(e)}"

def intelligent_scoring(text, criteria_name, nlp_processor):
    """Optimized intelligent scoring"""
    
    criteria_patterns = {
        "Data Requirements": {
            "essential_terms": ["data", "dataset", "training", "database", "information"],
            "positive_indicators": ["quality data", "structured", "available", "accessible"],
            "negative_indicators": ["no data", "limited", "poor quality", "unstructured"]
        },
        "Problem Complexity": {
            "essential_terms": ["complex", "advanced", "intelligent", "automated", "smart"],
            "positive_indicators": ["machine learning", "AI suitable", "optimization", "predictive"],
            "negative_indicators": ["simple", "basic", "manual", "straightforward"]
        },
        "Business Impact": {
            "essential_terms": ["business", "revenue", "cost", "efficiency", "value"],
            "positive_indicators": ["competitive advantage", "ROI", "strategic", "growth"],
            "negative_indicators": ["limited impact", "marginal", "nice to have"]
        },
        "Technical Feasibility": {
            "essential_terms": ["feasible", "technology", "platform", "integration", "implementation"],
            "positive_indicators": ["proven", "scalable", "maintainable", "robust"],
            "negative_indicators": ["challenging", "complex", "constraints", "limitations"]
        },
        "Timeline Alignment": {
            "essential_terms": ["timeline", "deadline", "schedule", "time", "priority"],
            "positive_indicators": ["reasonable", "adequate", "phased", "realistic"],
            "negative_indicators": ["tight", "unrealistic", "rushed", "immediate"]
        }
    }
    
    if criteria_name not in criteria_patterns:
        return 3, "No pattern defined for this criteria"
    
    pattern = criteria_patterns[criteria_name]
    text_lower = text.lower()
    
    # Fast scoring calculation
    essential_matches = sum(1 for term in pattern["essential_terms"] if term in text_lower)
    positive_matches = sum(1 for term in pattern["positive_indicators"] if term in text_lower)
    negative_matches = sum(1 for term in pattern["negative_indicators"] if term in text_lower)
    
    # Calculate score (1-5 range)
    base_score = min(5, max(1, 
        1 + (essential_matches * 0.8) + (positive_matches * 0.6) - (negative_matches * 0.4)
    ))
    
    final_score = round(base_score)
    
    # Generate concise reasoning
    reasoning_parts = []
    if essential_matches > 0:
        reasoning_parts.append(f"✓ Found {essential_matches} key terms")
    if positive_matches > 0:
        reasoning_parts.append(f"✓ {positive_matches} positive indicators")
    if negative_matches > 0:
        reasoning_parts.append(f"⚠ {negative_matches} concerns noted")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Limited relevant content"
    
    return final_score, reasoning

def create_score_card(title, percentage, description, reasoning, weight):
    """Create a beautiful score card with badges"""
    # Determine color class and badge based on percentage
    if percentage >= 80:
        color_class = "score-excellent"
        status = "Excellent"
        badge_class = "badge-excellent"
        emoji = "🟢"
    elif percentage >= 60:
        color_class = "score-good"
        status = "Good"
        badge_class = "badge-good"
        emoji = "🔵"
    elif percentage >= 40:
        color_class = "score-average"
        status = "Average"
        badge_class = "badge-average"
        emoji = "🟡"
    else:
        color_class = "score-poor"
        status = "Needs Attention"
        badge_class = "badge-poor"
        emoji = "🔴"
    
    card_html = f"""
    <div class="metric-card {color_class}">
        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.8rem;">
            <div>
                <h4 style="margin: 0; color: #2c3e50; display: flex; align-items: center; gap: 0.5rem;">
                    {emoji} {title}
                    <span class="score-badge {badge_class}">{status}</span>
                </h4>
                <div style="margin-top: 0.5rem;">
                    <span style="background: rgba(102, 126, 234, 0.1); padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; color: #667eea;">
                        Weight: {weight}%
                    </span>
                </div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 2rem; font-weight: bold; color: #2c3e50;">{percentage}%</span>
            </div>
        </div>
        <p style="color: #6c757d; font-size: 0.9rem; margin: 0.5rem 0; line-height: 1.4;">{description}</p>
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin-top: 0.8rem;">
            <p style="color: #495057; font-size: 0.85rem; margin: 0; font-style: italic;">📝 {reasoning}</p>
        </div>
    </div>
    """
    return card_html

# Streamlit App Configuration
st.set_page_config(
    page_title="AI Fit Scoring Dashboard",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state FIRST - before any other code
if 'scores' not in st.session_state:
    st.session_state.scores = {}
if 'reasoning' not in st.session_state:
    st.session_state.reasoning = {}
if 'nlp_analysis' not in st.session_state:
    st.session_state.nlp_analysis = {}
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'project_history' not in st.session_state:
    st.session_state.project_history = []
if 'selected_template' not in st.session_state:
    st.session_state.selected_template = "Custom Analysis"

# Define scoring criteria EARLY - so it's available throughout the script
criteria = {
    "Data Requirements": {
        "description": "Availability and quality of data needed for AI implementation",
        "weight": 25
    },
    "Problem Complexity": {
        "description": "How well-suited the problem is for AI/ML solutions", 
        "weight": 20
    },
    "Business Impact": {
        "description": "Potential value and ROI of implementing AI solution",
        "weight": 20
    },
    "Technical Feasibility": {
        "description": "Technical constraints and implementation challenges",
        "weight": 20
    },
    "Timeline Alignment": {
        "description": "Compatibility with project timelines and expectations",
        "weight": 15
    }
}

# Dashboard Header
st.markdown("""
<div class="dashboard-header">
    <h1>🎯 AI Fit Scoring Dashboard</h1>
    <p>Advanced project evaluation using NLP-powered analysis</p>
</div>
""", unsafe_allow_html=True)

# Initialize NLP processor (moved after session state init to prevent conflicts)
@st.cache_resource
def initialize_nlp():
    """Initialize NLP processor with caching"""
    return NLPProcessor()

nlp_processor = initialize_nlp()

# Sidebar for method selection and advanced features
st.sidebar.title("📋 Control Panel")

# Advanced Features Section
st.sidebar.markdown("### 🚀 Advanced Features")

# Project Templates
with st.sidebar.expander("📋 Project Templates"):
    template_options = {
        "Custom Analysis": "Use your own project details",
        "E-commerce Platform": "Online retail and recommendation systems",
        "Healthcare Analytics": "Patient data analysis and diagnostics", 
        "Financial Services": "Risk assessment and fraud detection",
        "Manufacturing Automation": "Process optimization and quality control",
        "Marketing Intelligence": "Customer segmentation and campaign optimization",
        "Supply Chain Optimization": "Logistics and inventory management"
    }
    
    selected_template = st.selectbox("Choose a template:", list(template_options.keys()), key="template_selector")
    
    if selected_template != "Custom Analysis":
        st.info(f"📝 {template_options[selected_template]}")
        if st.button("Apply Template", use_container_width=True, key="apply_template_btn"):
            st.session_state.selected_template = selected_template
            st.success(f"✅ Applied {selected_template} template")

# Comparative Analysis
with st.sidebar.expander("📊 Comparative Analysis"):
    st.markdown("**Compare Multiple Projects:**")
    
    if 'project_history' not in st.session_state:
        st.session_state.project_history = []
    
    if st.session_state.scores:
        project_name = st.text_input("Project Name:", placeholder="e.g., Customer Analytics v2", key="project_name_input")
        if st.button("💾 Save Current Analysis", use_container_width=True, key="save_analysis_btn"):
            if project_name:
                current_analysis = {
                    'name': project_name,
                    'scores': st.session_state.scores.copy(),
                    'overall_score': sum([(st.session_state.scores[name] - 1) / 4 * 100 * criteria[name]['weight'] / 100 
                                        for name in criteria.keys() if name in st.session_state.scores]),
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
                }
                st.session_state.project_history.append(current_analysis)
                st.success("✅ Project saved!")
            else:
                st.error("Please enter a project name")
    
    if st.session_state.project_history:
        st.write(f"**Saved Projects: {len(st.session_state.project_history)}**")
        for i, project in enumerate(st.session_state.project_history[-3:]):  # Show last 3
            st.write(f"• {project['name']}: {project['overall_score']:.0f}%")

# Industry Benchmarks
with st.sidebar.expander("🏭 Industry Benchmarks"):
    industry_benchmarks = {
        "Technology": {"avg_score": 75, "description": "Software & IT companies"},
        "Healthcare": {"avg_score": 68, "description": "Medical & pharmaceutical"},
        "Finance": {"avg_score": 72, "description": "Banking & financial services"},
        "Retail": {"avg_score": 65, "description": "E-commerce & retail"},
        "Manufacturing": {"avg_score": 62, "description": "Industrial & production"},
        "Education": {"avg_score": 58, "description": "Academic institutions"}
    }
    
    selected_industry = st.selectbox("Select Industry:", list(industry_benchmarks.keys()), key="industry_selector")
    
    if st.session_state.scores:
        current_overall = sum([(st.session_state.scores[name] - 1) / 4 * 100 * criteria[name]['weight'] / 100 
                             for name in criteria.keys() if name in st.session_state.scores])
        industry_avg = industry_benchmarks[selected_industry]["avg_score"]
        
        performance = "Above" if current_overall > industry_avg else "Below" if current_overall < industry_avg else "At"
        st.metric(
            f"{selected_industry} Avg", 
            f"{industry_avg}%", 
            f"{performance} average" if performance != "At" else "At average"
        )

# NLP Status Dashboard in Sidebar
with st.sidebar.expander("🔧 System Status"):
    statuses = [
        ("spaCy NLP", SPACY_AVAILABLE and nlp is not None),
        ("scikit-learn", SKLEARN_AVAILABLE),
        ("Transformers", TRANSFORMERS_AVAILABLE),
        ("Sentiment Analysis", SENTIMENT_AVAILABLE),
        ("PDF Processing", PDF_AVAILABLE),
        ("DOCX Processing", DOCX_AVAILABLE)
    ]
    
    for name, available in statuses:
        status_class = "status-available" if available else "status-unavailable"
        icon = "✅" if available else "❌"
        st.markdown(f'<span class="status-indicator {status_class}">{icon} {name}</span>', unsafe_allow_html=True)

input_method = st.sidebar.radio(
    "Input Method:",
    ["📄 Upload Document", "✍️ Manual Entry"],
    key="input_method_selector"
)

input_method = st.sidebar.radio(
    "Input Method:",
    ["📄 Upload Document", "✍️ Manual Entry"]
)

# Document Upload Section
if input_method == "📄 Upload Document":
    st.subheader("📤 Document Upload & Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload your project document",
        type=['pdf', 'docx', 'txt', 'md'],
        help="Supported formats: PDF, DOCX, TXT, MD (Max 200MB)",
        key="document_uploader"
    )
    
    if uploaded_file is not None:
        start_time = time.time()
        
        # File info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"✅ {uploaded_file.name}")
        with col2:
            st.info(f"📊 {uploaded_file.size / 1024:.1f} KB")
        
        # Extract text from file
        with st.spinner("🔄 Processing document..."):
            success, extracted_text, error_message = extract_text_from_file(uploaded_file)
        
        if success and extracted_text.strip():
            # Document Statistics
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)
            project_type = nlp_processor.extract_project_type(extracted_text)
            
            with col3:
                st.metric("📝 Words", f"{word_count:,}")
            
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("🔤 Characters", f"{char_count:,}")
            with col5:
                st.metric("🎯 Project Type", project_type)
            with col6:
                st.metric("⚡ Processing", f"{st.session_state.processing_time:.2f}s")
            
            # Action Buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("🤖 NLP Smart Analysis", type="primary", use_container_width=True, key="nlp_smart_analysis_btn"):
                    start_analysis = time.time()
                    
                    with st.spinner("🔍 Running NLP analysis..."):
                        progress_bar = st.progress(0)
                        
                        # Process each criteria with progress updates
                        for i, criteria_name in enumerate(criteria.keys()):
                            progress_bar.progress((i + 1) / len(criteria))
                            score, reasoning = intelligent_scoring(extracted_text, criteria_name, nlp_processor)
                            st.session_state.scores[criteria_name] = score
                            st.session_state.reasoning[criteria_name] = reasoning
                        
                        # Store analysis results
                        st.session_state.nlp_analysis = {
                            'project_type': project_type,
                            'word_count': word_count,
                            'entities': nlp_processor.extract_entities(extracted_text),
                            'keywords': nlp_processor.extract_keywords(extracted_text),
                            'sentiment': nlp_processor.analyze_sentiment(extracted_text)
                        }
                        
                        progress_bar.progress(100)
                        st.session_state.processing_time = time.time() - start_analysis
                    
                    st.success(f"🎯 Analysis completed in {st.session_state.processing_time:.2f} seconds!")
                    st.rerun()
            
            with col2:
                with st.expander("📖 Document Preview"):
                    st.text_area("Content Preview:", 
                               extracted_text[:800] + "..." if len(extracted_text) > 800 else extracted_text,
                               height=200, disabled=True)
            
            with col3:
                if st.session_state.nlp_analysis:
                    with st.expander("🧠 NLP Insights"):
                        nlp_data = st.session_state.nlp_analysis
                        
                        if nlp_data.get('keywords'):
                            st.write("**🔑 Top Keywords:**")
                            for keyword, score in nlp_data['keywords'][:5]:
                                st.write(f"• {keyword} ({score:.3f})")
                        
                        if nlp_data.get('sentiment'):
                            sentiment = nlp_data['sentiment']
                            st.write(f"**😊 Sentiment:** {sentiment['label']} ({sentiment['confidence']:.2f})")
            
            with col4:
                if st.button("🔄 Reset Analysis", use_container_width=True):
                    st.session_state.scores = {}
                    st.session_state.reasoning = {}
                    st.session_state.nlp_analysis = {}
                    st.rerun()
                    
        else:
            st.error(f"❌ Error: {error_message}")

# Manual Entry Section
elif input_method == "✍️ Manual Entry":
    st.subheader("✍️ Manual Project Entry")
    project_description = st.text_area(
        "Project Description",
        height=150,
        placeholder="Describe your project, requirements, and objectives in detail...",
        key="manual_project_description"
    )
    
    if project_description and len(project_description) > 50:
        if st.button("🧠 Analyze Description", type="primary", key="analyze_description_btn"):
            start_analysis = time.time()
            with st.spinner("🔍 Analyzing description..."):
                for criteria_name in criteria.keys():
                    score, reasoning = intelligent_scoring(project_description, criteria_name, nlp_processor)
                    st.session_state.scores[criteria_name] = score
                    st.session_state.reasoning[criteria_name] = reasoning
                
                st.session_state.processing_time = time.time() - start_analysis
            
            st.success(f"✨ Analysis completed in {st.session_state.processing_time:.2f} seconds!")
            st.rerun()

# Results Dashboard
if st.session_state.scores:
    st.markdown("---")
    st.subheader("📊 AI Fit Assessment Dashboard")
    
    # Calculate overall score
    total_weighted_score = 0
    total_weight = 0
    
    for criteria_name, criteria_info in criteria.items():
        if criteria_name in st.session_state.scores:
            score = st.session_state.scores[criteria_name]  # 1-5 scale
            percentage = (score - 1) / 4 * 100  # Convert to 0-100%
            weight = criteria_info["weight"]
            total_weighted_score += percentage * (weight / 100)
            total_weight += weight
    
    overall_percentage = total_weighted_score if total_weight > 0 else 0
    
    # Overall Score Card
    if overall_percentage >= 80:
        status_color = "#00C851"
        status_text = "🟢 Excellent AI Fit - Highly Recommended"
        recommendation = "This project shows exceptional potential for AI implementation. Proceed with confidence."
    elif overall_percentage >= 60:
        status_color = "#39C0ED"
        status_text = "🔵 Good AI Fit - Recommended"
        recommendation = "This project has good AI potential. Consider moving forward with proper planning."
    elif overall_percentage >= 40:
        status_color = "#ffbb33"
        status_text = "🟡 Moderate AI Fit - Proceed with Caution"
        recommendation = "This project has moderate AI potential. Careful evaluation and risk mitigation recommended."
    else:
        status_color = "#FF4444"
        status_text = "🔴 Low AI Fit - Not Recommended"
        recommendation = "This project shows limited AI potential. Consider alternative approaches."
    
    # Main Score Display
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {status_color}15, {status_color}05); 
                padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;
                border: 2px solid {status_color}30;">
        <h2 style="color: {status_color}; margin: 0;">Overall AI Fit Score</h2>
        <div style="font-size: 4rem; font-weight: bold; color: {status_color}; margin: 1rem 0;">
            {overall_percentage:.0f}%
        </div>
        <h3 style="color: #2c3e50; margin: 0.5rem 0;">{status_text}</h3>
        <p style="color: #6c757d; font-size: 1.1rem; margin: 0;">{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Score Breakdown
    st.subheader("📋 Detailed Score Breakdown")
    
    # Create score cards in two columns
    col1, col2 = st.columns(2)
    
    criteria_items = list(criteria.items())
    for i, (criteria_name, criteria_info) in enumerate(criteria_items):
        if criteria_name in st.session_state.scores:
            score = st.session_state.scores[criteria_name]
            percentage = round((score - 1) / 4 * 100)
            reasoning = st.session_state.reasoning.get(criteria_name, "No reasoning provided")
            
            card_html = create_score_card(
                criteria_name, 
                percentage, 
                criteria_info["description"], 
                reasoning,
                criteria_info["weight"]
            )
            
            # Alternate between columns
            if i % 2 == 0:
                with col1:
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                with col2:
                    st.markdown(card_html, unsafe_allow_html=True)
    
    # Visualization Section
    st.subheader("📈 Visual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar Chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        criteria_names = list(st.session_state.scores.keys())
        scores_percentage = [(st.session_state.scores[name] - 1) / 4 * 100 for name in criteria_names]
        
        # Create angles for each criteria
        angles = np.linspace(0, 2 * np.pi, len(criteria_names), endpoint=False).tolist()
        
        # Complete the circle by adding the first point
        scores_percentage_plot = scores_percentage + [scores_percentage[0]]
        angles_plot = angles + [angles[0]]
        
        # Plot the radar chart
        ax.plot(angles_plot, scores_percentage_plot, 'o-', linewidth=2, color='#667eea')
        ax.fill(angles_plot, scores_percentage_plot, alpha=0.25, color='#667eea')
        
        # Set the labels and ticks
        ax.set_xticks(angles)
        ax.set_xticklabels([name.replace(' ', '\n') for name in criteria_names])
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.set_title("AI Fit Score Profile", size=16, pad=20)
        ax.grid(True)
        
        st.pyplot(fig)
    
    with col2:
        # Bar Chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use original scores_percentage (not the one modified for radar chart)
        original_scores_percentage = [(st.session_state.scores[name] - 1) / 4 * 100 for name in criteria_names]
        
        y_pos = np.arange(len(criteria_names))
        colors = ['#00C851' if score >= 80 else '#39C0ED' if score >= 60 else '#ffbb33' if score >= 40 else '#FF4444' 
                 for score in original_scores_percentage]
        
        bars = ax.barh(y_pos, original_scores_percentage, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.replace(' ', '\n') for name in criteria_names])
        ax.set_xlabel('Score (%)')
        ax.set_title('Detailed Score Breakdown', size=16, pad=20)
        ax.set_xlim(0, 100)
        
        # Add percentage labels on bars
        for bar, score in zip(bars, original_scores_percentage):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                   f'{score:.0f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Export Section with Comparative Analysis
    st.subheader("📤 Export & Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Prepare detailed report data
        report_data = []
        for criteria_name in criteria.keys():
            if criteria_name in st.session_state.scores:
                score = st.session_state.scores[criteria_name]
                percentage = round((score - 1) / 4 * 100)
                report_data.append({
                    'Criteria': criteria_name,
                    'Score (1-5)': score,
                    'Percentage': f"{percentage}%",
                    'Weight': f"{criteria[criteria_name]['weight']}%",
                    'Weighted Score': f"{percentage * criteria[criteria_name]['weight'] / 100:.1f}%",
                    'Status': 'Excellent' if percentage >= 80 else 'Good' if percentage >= 60 else 'Average' if percentage >= 40 else 'Poor',
                    'Reasoning': st.session_state.reasoning.get(criteria_name, ""),
                    'Description': criteria[criteria_name]['description']
                })
        
        # Add overall summary
        summary_row = {
            'Criteria': 'OVERALL ASSESSMENT',
            'Score (1-5)': f"{1 + (overall_percentage / 25):.1f}",
            'Percentage': f"{overall_percentage:.0f}%",
            'Weight': "100%",
            'Weighted Score': f"{overall_percentage:.0f}%",
            'Status': status_text.split(' - ')[0].replace('🟢 ', '').replace('🔵 ', '').replace('🟡 ', '').replace('🔴 ', ''),
            'Reasoning': recommendation,
            'Description': 'Complete AI Fit Assessment'
        }
        
        if report_data:
            df = pd.DataFrame(report_data)
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
            
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Report",
                data=csv,
                file_name=f"ai_fit_assessment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_report_btn"
            )
    
    with col2:
        st.metric("📊 Processing Time", f"{st.session_state.processing_time:.2f}s")
        st.metric("🎯 Confidence", "High" if len(st.session_state.scores) == len(criteria) else "Moderate")
    
    with col3:
        if st.session_state.nlp_analysis:
            nlp_data = st.session_state.nlp_analysis
            st.metric("📝 Words Analyzed", f"{nlp_data.get('word_count', 0):,}")
            st.metric("🏷️ Entities Found", len(nlp_data.get('entities', [])))

# Comparative Analysis Dashboard
if len(st.session_state.project_history) > 1:
    st.markdown("---")
    st.subheader("📊 Project Comparison Dashboard")
    
    # Create comparison dataframe
    comparison_data = []
    for project in st.session_state.project_history[-5:]:  # Last 5 projects
        comparison_data.append({
            'Project': project['name'],
            'Overall Score': f"{project['overall_score']:.0f}%",
            'Date': project['timestamp'],
            **{criteria_name: f"{((project['scores'].get(criteria_name, 3) - 1) / 4 * 100):.0f}%" 
               for criteria_name in criteria.keys()}
        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Comparison visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall scores comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            project_names = [p['name'] for p in st.session_state.project_history[-5:]]
            overall_scores = [p['overall_score'] for p in st.session_state.project_history[-5:]]
            
            bars = ax.bar(range(len(project_names)), overall_scores, 
                         color=['#28a745' if score >= 60 else '#ffc107' if score >= 40 else '#dc3545' 
                               for score in overall_scores], alpha=0.8)
            
            ax.set_xlabel('Projects')
            ax.set_ylabel('Overall AI Fit Score (%)')
            ax.set_title('Project Comparison - Overall Scores')
            ax.set_xticks(range(len(project_names)))
            ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                               for name in project_names], rotation=45, ha='right')
            ax.set_ylim(0, 100)
            
            # Add score labels on bars
            for bar, score in zip(bars, overall_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{score:.0f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Criteria comparison for latest project vs average
            if st.session_state.scores:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                criteria_names_short = [name.replace(' ', '\n') for name in criteria.keys()]
                current_scores = [(st.session_state.scores.get(name, 3) - 1) / 4 * 100 
                                for name in criteria.keys()]
                
                # Calculate average scores across all historical projects
                avg_scores = []
                for criteria_name in criteria.keys():
                    scores = [((p['scores'].get(criteria_name, 3) - 1) / 4 * 100) 
                             for p in st.session_state.project_history]
                    avg_scores.append(sum(scores) / len(scores) if scores else 0)
                
                x = np.arange(len(criteria_names_short))
                width = 0.35
                
                ax.bar(x - width/2, current_scores, width, label='Current Project', 
                      color='#667eea', alpha=0.8)
                ax.bar(x + width/2, avg_scores, width, label='Historical Average', 
                      color='#95a5a6', alpha=0.8)
                
                ax.set_xlabel('Criteria')
                ax.set_ylabel('Score (%)')
                ax.set_title('Current vs Historical Performance')
                ax.set_xticks(x)
                ax.set_xticklabels(criteria_names_short)
                ax.legend()
                ax.set_ylim(0, 100)
                
                plt.tight_layout()
                st.pyplot(fig)

else:
    # Enhanced welcome message showcasing new features using Streamlit components
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-bottom: 2rem;">
        <h3 style="color: #2c3e50; margin-bottom: 1rem;">👋 Welcome to AI Fit Scoring Dashboard</h3>
        <p style="font-size: 1.1rem; color: #6c757d; margin-bottom: 1rem;">
            Upload a document or enter project details to begin your comprehensive AI readiness assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase using Streamlit columns
    st.subheader("🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745; margin-bottom: 1rem;">
                <h4 style="color: #28a745; margin: 0 0 0.5rem 0;">🧠 Advanced NLP Analysis</h4>
                <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">
                    Powered by spaCy, Transformers, and scikit-learn for intelligent semantic understanding.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ffc107; margin-bottom: 1rem;">
                <h4 style="color: #e0a800; margin: 0 0 0.5rem 0;">📋 Industry Templates</h4>
                <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">
                    Pre-configured templates for different industries and use cases.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #17a2b8; margin-bottom: 1rem;">
                <h4 style="color: #17a2b8; margin: 0 0 0.5rem 0;">📊 Project Comparison</h4>
                <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">
                    Save and compare multiple projects with visual analytics and trend analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545; margin-bottom: 1rem;">
                <h4 style="color: #dc3545; margin: 0 0 0.5rem 0;">🏭 Industry Benchmarks</h4>
                <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">
                    Compare your projects against industry averages and best practices.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 15px;">
        <p style="color: white; font-size: 1.2rem; font-weight: 500; margin: 0;">
            🚀 Get started by uploading a document or entering project details manually!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #6c757d;">
    <p><strong>🎯 AI Fit Scoring Dashboard v2.0</strong></p>
    <p>Powered by spaCy • scikit-learn • Transformers • Streamlit</p>
</div>
""", unsafe_allow_html=True)