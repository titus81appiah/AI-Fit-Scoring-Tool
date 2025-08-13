import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re
import numpy as np
from collections import Counter
import warnings
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
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
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

class NLPProcessor:
    def __init__(self):
        self.nlp = nlp
        self.tfidf = None
        if SKLEARN_AVAILABLE:
            self.tfidf = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1
            )
    
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'[^\w\s\-\.]', ' ', text)  # Keep alphanumeric, spaces, hyphens, dots
        text = text.lower().strip()
        
        return text
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def extract_keywords(self, text, top_k=10):
        """Extract important keywords using TF-IDF"""
        if not SKLEARN_AVAILABLE or not text:
            return []
        
        try:
            processed_text = self.preprocess_text(text)
            
            # Fit TF-IDF
            tfidf_matrix = self.tfidf.fit_transform([processed_text])
            feature_names = self.tfidf.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = np.argsort(tfidf_scores)[::-1][:top_k]
            keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
            
            return keywords
        except:
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment and confidence"""
        if not SENTIMENT_AVAILABLE or not text:
            return None
        
        try:
            results = sentiment_pipeline(text[:512])  # Limit text length
            if results and len(results[0]) > 0:
                # Get the most confident prediction
                best_result = max(results[0], key=lambda x: x['score'])
                return {
                    'label': best_result['label'],
                    'confidence': best_result['score'],
                    'all_scores': results[0]
                }
        except:
            return None
        
        return None
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between texts"""
        if not SKLEARN_AVAILABLE:
            return 0.0
        
        try:
            processed_texts = [self.preprocess_text(text1), self.preprocess_text(text2)]
            tfidf_matrix = self.tfidf.fit_transform(processed_texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def extract_project_type(self, text):
        """Classify project type based on content analysis"""
        text_lower = text.lower()
        
        project_indicators = {
            'Web Application': ['web', 'website', 'webapp', 'frontend', 'backend', 'api', 'rest', 'http'],
            'Mobile Application': ['mobile', 'app', 'ios', 'android', 'smartphone', 'tablet'],
            'Data Analysis': ['data', 'analytics', 'dashboard', 'reporting', 'visualization', 'insights'],
            'Machine Learning': ['ml', 'ai', 'prediction', 'classification', 'regression', 'neural network'],
            'Automation': ['automate', 'workflow', 'process', 'efficiency', 'streamline', 'optimize'],
            'E-commerce': ['shop', 'store', 'payment', 'cart', 'checkout', 'inventory', 'catalog'],
            'Content Management': ['cms', 'content', 'blog', 'articles', 'publishing', 'editorial']
        }
        
        scores = {}
        for project_type, indicators in project_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[project_type] = score
        
        # Return top project type if score > 0
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "General Software"

def extract_text_from_file(uploaded_file):
    """Extract text from various file formats"""
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    try:
        if file_extension == 'txt':
            text = str(uploaded_file.read(), "utf-8")
            return True, text, ""
            
        elif file_extension == 'md':
            text = str(uploaded_file.read(), "utf-8")
            return True, text, ""
            
        elif file_extension == 'docx':
            if not DOCX_AVAILABLE:
                return False, "", "python-docx library not installed."
            
            doc = Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return True, text, ""
            
        elif file_extension == 'pdf':
            if not PDF_AVAILABLE:
                return False, "", "PDF processing library not installed."
            
            text = ""
            if 'USE_PDFPLUMBER' in globals() and USE_PDFPLUMBER:
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
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

def intelligent_scoring(text, criteria_name, nlp_processor):
    """Advanced NLP-based scoring for each criteria"""
    
    # Define sophisticated criteria patterns
    criteria_patterns = {
        "Data Requirements": {
            "essential_terms": ["data", "dataset", "training", "machine learning", "algorithm", "model"],
            "positive_indicators": ["big data", "quality data", "clean data", "structured data", "data pipeline", "data warehouse"],
            "negative_indicators": ["no data", "limited data", "poor quality", "unstructured", "messy data"],
            "context_weight": 0.4
        },
        "Problem Complexity": {
            "essential_terms": ["complex", "sophisticated", "advanced", "intelligent", "automated", "optimization"],
            "positive_indicators": ["AI suitable", "machine learning", "pattern recognition", "predictive", "adaptive"],
            "negative_indicators": ["simple", "basic", "straightforward", "manual", "rule-based", "static"],
            "context_weight": 0.3
        },
        "Business Impact": {
            "essential_terms": ["business", "revenue", "cost", "efficiency", "productivity", "ROI", "value"],
            "positive_indicators": ["competitive advantage", "strategic", "scalable", "growth", "transformation"],
            "negative_indicators": ["marginal", "limited impact", "nice to have", "small scale"],
            "context_weight": 0.5
        },
        "Technical Feasibility": {
            "essential_terms": ["feasible", "technology", "platform", "infrastructure", "API", "integration"],
            "positive_indicators": ["proven technology", "existing tools", "scalable", "maintainable", "robust"],
            "negative_indicators": ["challenging", "complex integration", "legacy system", "constraints", "limitations"],
            "context_weight": 0.3
        },
        "Timeline Alignment": {
            "essential_terms": ["timeline", "deadline", "schedule", "priority", "urgent", "time"],
            "positive_indicators": ["reasonable timeline", "adequate time", "phased approach", "iterative"],
            "negative_indicators": ["tight deadline", "unrealistic", "rushed", "no time", "immediate"],
            "context_weight": 0.2
        }
    }
    
    if criteria_name not in criteria_patterns:
        return 3, "No pattern defined for this criteria"
    
    pattern = criteria_patterns[criteria_name]
    text_lower = text.lower()
    
    # Calculate different scoring components
    essential_score = 0
    positive_score = 0
    negative_score = 0
    
    # Count essential terms
    essential_matches = [term for term in pattern["essential_terms"] if term in text_lower]
    essential_score = min(len(essential_matches) / len(pattern["essential_terms"]) * 3, 3)
    
    # Count positive indicators
    positive_matches = [term for term in pattern["positive_indicators"] if term in text_lower]
    positive_score = min(len(positive_matches) * 0.5, 2)
    
    # Count negative indicators (subtract from score)
    negative_matches = [term for term in pattern["negative_indicators"] if term in text_lower]
    negative_score = min(len(negative_matches) * 0.3, 1.5)
    
    # Extract keywords using NLP
    keywords = nlp_processor.extract_keywords(text, top_k=20)
    relevant_keywords = [kw[0] for kw in keywords if any(term in kw[0] for term in pattern["essential_terms"])]
    
    # Sentiment analysis for context
    sentiment = nlp_processor.analyze_sentiment(text)
    sentiment_boost = 0
    if sentiment:
        if sentiment['label'] in ['LABEL_2', 'POSITIVE'] and sentiment['confidence'] > 0.7:
            sentiment_boost = 0.5
        elif sentiment['label'] in ['LABEL_0', 'NEGATIVE'] and sentiment['confidence'] > 0.7:
            sentiment_boost = -0.5
    
    # Calculate final score
    base_score = essential_score + positive_score - negative_score + sentiment_boost
    final_score = max(1, min(5, round(base_score + 1)))  # Ensure 1-5 range
    
    # Generate detailed reasoning
    reasoning_parts = []
    if essential_matches:
        reasoning_parts.append(f"✓ Found key terms: {', '.join(essential_matches[:3])}")
    if positive_matches:
        reasoning_parts.append(f"✓ Positive indicators: {', '.join(positive_matches[:2])}")
    if negative_matches:
        reasoning_parts.append(f"⚠ Concerns: {', '.join(negative_matches[:2])}")
    if relevant_keywords:
        reasoning_parts.append(f"✓ Relevant keywords: {', '.join(relevant_keywords[:3])}")
    if sentiment and sentiment['confidence'] > 0.6:
        reasoning_parts.append(f"📊 Sentiment: {sentiment['label']} ({sentiment['confidence']:.2f})")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Limited relevant content found"
    
    return final_score, reasoning

# Streamlit App Configuration
st.set_page_config(
    page_title="AI Fit Scoring Tool",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Fit Scoring Tool with NLP Analysis")
st.markdown("Advanced project evaluation using Natural Language Processing")

# Display NLP capabilities status
with st.expander("🔧 NLP Capabilities Status"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("📊 **Core NLP:**")
        st.write(f"• spaCy: {'✅' if SPACY_AVAILABLE and nlp else '❌'}")
        st.write(f"• scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    with col2:
        st.write("🧠 **Advanced NLP:**")
        st.write(f"• Transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
        st.write(f"• Sentiment Analysis: {'✅' if SENTIMENT_AVAILABLE else '❌'}")
    with col3:
        st.write("📄 **Document Processing:**")
        st.write(f"• PDF: {'✅' if PDF_AVAILABLE else '❌'}")
        st.write(f"• DOCX: {'✅' if DOCX_AVAILABLE else '❌'}")

# Initialize NLP processor
nlp_processor = NLPProcessor()

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
if 'nlp_analysis' not in st.session_state:
    st.session_state.nlp_analysis = {}

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
    st.header("📤 Document Upload & NLP Analysis")
    
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
        
        if success and extracted_text.strip():
            # Show document statistics
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", f"{word_count:,}")
            with col2:
                st.metric("Characters", f"{char_count:,}")
            with col3:
                project_type = nlp_processor.extract_project_type(extracted_text)
                st.metric("Detected Type", project_type)
            
            # Show extracted text preview
            with st.expander("📖 Document Preview"):
                st.text_area("Extracted Text Preview:", 
                           extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
                           height=200)
            
            # NLP Analysis Section
            if SPACY_AVAILABLE or SKLEARN_AVAILABLE:
                with st.expander("🧠 NLP Analysis Results"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if nlp_processor.nlp:
                            entities = nlp_processor.extract_entities(extracted_text)
                            if entities:
                                st.subheader("🏷️ Named Entities")
                                entity_df = pd.DataFrame(entities)
                                st.dataframe(entity_df, use_container_width=True)
                        
                        if SKLEARN_AVAILABLE:
                            keywords = nlp_processor.extract_keywords(extracted_text, top_k=15)
                            if keywords:
                                st.subheader("🔑 Key Terms (TF-IDF)")
                                for keyword, score in keywords[:10]:
                                    st.write(f"• **{keyword}** ({score:.3f})")
                    
                    with col2:
                        if SENTIMENT_AVAILABLE:
                            sentiment = nlp_processor.analyze_sentiment(extracted_text)
                            if sentiment:
                                st.subheader("😊 Sentiment Analysis")
                                st.write(f"**Overall:** {sentiment['label']}")
                                st.write(f"**Confidence:** {sentiment['confidence']:.2f}")
                                
                                # Show all scores
                                for score_info in sentiment['all_scores']:
                                    st.progress(score_info['score'], text=f"{score_info['label']}: {score_info['score']:.2f}")
            
            # Auto-scoring options
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🤖 NLP Smart Scoring", type="primary"):
                    with st.spinner("🔍 Analyzing with NLP..."):
                        for criteria_name in criteria.keys():
                            score, reasoning = intelligent_scoring(extracted_text, criteria_name, nlp_processor)
                            st.session_state.scores[criteria_name] = score
                            st.session_state.reasoning[criteria_name] = reasoning
                        
                        # Store NLP analysis
                        st.session_state.nlp_analysis = {
                            'project_type': project_type,
                            'word_count': word_count,
                            'entities': nlp_processor.extract_entities(extracted_text) if nlp_processor.nlp else [],
                            'keywords': nlp_processor.extract_keywords(extracted_text, top_k=10) if SKLEARN_AVAILABLE else [],
                            'sentiment': nlp_processor.analyze_sentiment(extracted_text) if SENTIMENT_AVAILABLE else None
                        }
                    
                    st.success("🎯 NLP-powered scoring completed!")
                    st.rerun()
            
            with col2:
                if st.button("📊 Basic Auto-Score"):
                    # Fallback to simple keyword-based scoring if NLP not available
                    with st.spinner("📊 Basic analysis..."):
                        for criteria_name in criteria.keys():
                            score, reasoning = intelligent_scoring(extracted_text, criteria_name, nlp_processor)
                            st.session_state.scores[criteria_name] = max(1, score - 1)  # More conservative
                            st.session_state.reasoning[criteria_name] = f"Basic analysis: {reasoning}"
                    
                    st.success("📊 Basic scoring completed!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Reset All"):
                    st.session_state.scores = {}
                    st.session_state.reasoning = {}
                    st.session_state.nlp_analysis = {}
                    st.rerun()
                    
        else:
            st.error(f"❌ Error processing file: {error_message}")

# Manual Entry Section
elif input_method == "✍️ Manual Entry":
    st.header("✍️ Manual Project Entry")
    project_description = st.text_area(
        "Project Description",
        height=150,
        placeholder="Describe your project, requirements, and objectives..."
    )
    
    if project_description and len(project_description) > 50:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Analyze Text with NLP", type="primary"):
                with st.spinner("🔍 Analyzing text..."):
                    for criteria_name in criteria.keys():
                        score, reasoning = intelligent_scoring(project_description, criteria_name, nlp_processor)
                        st.session_state.scores[criteria_name] = score
                        st.session_state.reasoning[criteria_name] = reasoning
                
                st.success("✨ NLP analysis completed!")
                st.rerun()

# Scoring Interface
st.header("📊 AI Fit Scoring Dashboard")

col1, col2 = st.columns([2, 1])

with col1:
    for criteria_name, criteria_info in criteria.items():
        st.subheader(f"{criteria_name} ({criteria_info['weight']*100:.0f}%)")
        st.caption(criteria_info["description"])
        
        # Get current score from session state
        current_score = st.session_state.scores.get(criteria_name, 3)
        current_reasoning = st.session_state.reasoning.get(criteria_name, "")
        
        # Score slider with better labels
        score = st.select_slider(
            f"Score for {criteria_name}",
            options=[1, 2, 3, 4, 5],
            value=current_score,
            format_func=lambda x: f"{x} - {'Poor' if x==1 else 'Below Average' if x==2 else 'Average' if x==3 else 'Good' if x==4 else 'Excellent'}",
            key=f"score_{criteria_name}"
        )
        
        # Reasoning text area with NLP insights
        reasoning = st.text_area(
            f"Reasoning for {criteria_name}",
            value=current_reasoning,
            height=120,
            key=f"reasoning_{criteria_name}",
            placeholder="Explain your scoring rationale... (Auto-populated if using NLP analysis)"
        )
        
        # Update session state
        st.session_state.scores[criteria_name] = score
        st.session_state.reasoning[criteria_name] = reasoning
        
        st.divider()

# Enhanced Results Panel
with col2:
    st.subheader("📈 Smart Results Dashboard")
    
    # Show NLP insights if available
    if st.session_state.nlp_analysis:
        st.subheader("🧠 NLP Insights")
        nlp_data = st.session_state.nlp_analysis
        
        if nlp_data.get('project_type'):
            st.info(f"**Project Type:** {nlp_data['project_type']}")
        
        if nlp_data.get('sentiment'):
            sentiment = nlp_data['sentiment']
            st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['confidence']:.2f})")
    
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
            
            # Display overall score with enhanced styling
            st.metric("🎯 Overall AI Fit Score", f"{weighted_average:.2f}/5.00")
            
            # Enhanced recommendation with confidence
            confidence = "High" if len(st.session_state.scores) == len(criteria) else "Moderate"
            
            if weighted_average >= 4.0:
                st.success(f"🟢 **Excellent AI Fit** - Strongly recommended ({confidence} confidence)")
            elif weighted_average >= 3.5:
                st.success(f"🟢 **Good AI Fit** - Recommended ({confidence} confidence)")
            elif weighted_average >= 2.5:
                st.warning(f"🟡 **Moderate AI Fit** - Consider with caution ({confidence} confidence)")
            else:
                st.error(f"🔴 **Poor AI Fit** - Not recommended ({confidence} confidence)")
            
            # Detailed score breakdown
            st.subheader("📊 Score Breakdown")
            for criteria_name in criteria.keys():
                if criteria_name in st.session_state.scores:
                    score = st.session_state.scores[criteria_name]
                    weight = criteria[criteria_name]["weight"]
                    weighted_score = score * weight
                    st.write(f"**{criteria_name}:** {score}/5 (weighted: {weighted_score:.2f})")
            
            # Enhanced visualization
            if len(st.session_state.scores) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Bar chart
                criteria_names = list(st.session_state.scores.keys())
                scores = list(st.session_state.scores.values())
                
                bars = ax1.barh(criteria_names, scores, color='skyblue', alpha=0.8)
                ax1.set_xlim(0, 5)
                ax1.set_xlabel('Score')
                ax1.set_title('AI Fit Scoring Results')
                
                # Add score labels
                for bar, score in zip(bars, scores):
                    ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                           str(score), va='center', fontweight='bold')
                
                # Pie chart for weighted contribution
                weights = [criteria[name]["weight"] * st.session_state.scores[name] for name in criteria_names]
                ax2.pie(weights, labels=criteria_names, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Weighted Score Distribution')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Export enhanced results
                st.subheader("📤 Export Results")
                
                # Create comprehensive report
                report_data = []
                for criteria_name in criteria.keys():
                    if criteria_name in st.session_state.scores:
                        report_data.append({
                            'Criteria': criteria_name,
                            'Score': st.session_state.scores[criteria_name],
                            'Weight': criteria[criteria_name]['weight'],
                            'Weighted Score': st.session_state.scores[criteria_name] * criteria[criteria_name]['weight'],
                            'Reasoning': st.session_state.reasoning.get(criteria_name, ""),
                            'Description': criteria[criteria_name]['description']
                        })
                
                if report_data:
                    df = pd.DataFrame(report_data)
                    
                    # Add summary row
                    summary_row = {
                        'Criteria': 'OVERALL SCORE',
                        'Score': f"{weighted_average:.2f}/5.00",
                        'Weight': 1.0,
                        'Weighted Score': weighted_average,
                        'Reasoning': f"{'Recommended' if weighted_average >= 3.0 else 'Not Recommended'}",
                        'Description': 'Final AI Fit Assessment'
                    }
                    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
                    
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Detailed Report (CSV)",
                        data=csv,
                        file_name=f"ai_fit_scoring_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

# Footer with enhanced info
st.markdown("---")
st.markdown("**🤖 AI Fit Scoring Tool v2.0** - Advanced NLP-powered project evaluation")
st.caption("Powered by spaCy, scikit-learn, and Transformers for intelligent analysis")