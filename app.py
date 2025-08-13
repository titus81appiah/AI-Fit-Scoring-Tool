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

# NLP imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

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
if 'nlp_analysis' not in st.session_state:
    st.session_state.nlp_analysis = {}

# Define criteria
criteria = {
    "Data Requirements": {"description": "Availability and quality of data needed for AI implementation", "weight": 25},
    "Problem Complexity": {"description": "How well-suited the problem is for AI/ML solutions", "weight": 20},
    "Business Impact": {"description": "Potential value and ROI of implementing AI solution", "weight": 20},
    "Technical Feasibility": {"description": "Technical constraints and implementation challenges", "weight": 20},
    "Timeline Alignment": {"description": "Compatibility with project timelines and expectations", "weight": 15}
}

class NLPAnalyzer:
    def __init__(self):
        self.tfidf = None
        if SKLEARN_AVAILABLE:
            self.tfidf = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
    
    def extract_keywords(self, text, top_k=10):
        """Extract important keywords using TF-IDF"""
        if not SKLEARN_AVAILABLE or not text:
            return []
        
        try:
            processed_text = self.preprocess_text(text)
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
        """Analyze sentiment using TextBlob"""
        if not TEXTBLOB_AVAILABLE or not text:
            return None
        
        try:
            blob = TextBlob(text[:1000])  # Limit text length
            polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
            subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            
            # Convert polarity to label
            if polarity > 0.1:
                label = "POSITIVE"
                confidence = 0.5 + (polarity * 0.5)
            elif polarity < -0.1:
                label = "NEGATIVE"
                confidence = 0.5 + (abs(polarity) * 0.5)
            else:
                label = "NEUTRAL"
                confidence = 0.5 + (0.5 - abs(polarity))
            
            return {
                'label': label,
                'confidence': min(confidence, 0.95),
                'polarity': polarity,
                'subjectivity': subjectivity
            }
        except:
            return None
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = text.lower().strip()
        
        return text
    
    def extract_project_insights(self, text):
        """Extract project-specific insights"""
        insights = {}
        text_lower = text.lower()
        
        # Technology detection
        tech_keywords = {
            'Python': ['python', 'django', 'flask', 'pandas', 'numpy'],
            'JavaScript': ['javascript', 'node.js', 'react', 'vue', 'angular'],
            'Machine Learning': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'neural network'],
            'Cloud': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes'],
            'Database': ['sql', 'mongodb', 'postgresql', 'mysql', 'database']
        }
        
        detected_tech = []
        for tech, keywords in tech_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected_tech.append(tech)
        
        insights['technologies'] = detected_tech
        
        # Timeline detection
        timeline_patterns = [
            r'(\d+)\s*(week|month|year)s?',
            r'(quarter|q[1-4])',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(\d{4})'  # Year
        ]
        
        timeline_mentions = []
        for pattern in timeline_patterns:
            matches = re.findall(pattern, text_lower)
            timeline_mentions.extend(matches)
        
        insights['timeline_mentions'] = timeline_mentions[:5]  # Limit to 5
        
        # Budget/cost detection
        budget_patterns = [
            r'\$\d+[,\d]*',
            r'\d+[,\d]*\s*dollars?',
            r'budget.*?\$?\d+[,\d]*',
            r'cost.*?\$?\d+[,\d]*'
        ]
        
        budget_mentions = []
        for pattern in budget_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            budget_mentions.extend(matches)
        
        insights['budget_mentions'] = budget_mentions[:3]  # Limit to 3
        
        return insights

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

def advanced_keyword_analysis(text, criteria_name, nlp_analyzer):
    """Advanced NLP-based scoring"""
    if not text:
        return 3, "No text provided"
    
    # Define sophisticated criteria patterns
    criteria_patterns = {
        "Data Requirements": {
            "essential_terms": ["data", "dataset", "training", "machine learning", "algorithm", "model"],
            "positive_indicators": ["quality data", "clean data", "structured data", "data pipeline", "big data"],
            "negative_indicators": ["no data", "limited data", "poor quality", "unstructured"]
        },
        "Problem Complexity": {
            "essential_terms": ["complex", "advanced", "intelligent", "automated", "optimization"],
            "positive_indicators": ["AI suitable", "machine learning", "pattern recognition", "predictive"],
            "negative_indicators": ["simple", "basic", "straightforward", "manual", "rule-based"]
        },
        "Business Impact": {
            "essential_terms": ["business", "revenue", "cost", "efficiency", "productivity", "ROI"],
            "positive_indicators": ["competitive advantage", "strategic", "scalable", "growth"],
            "negative_indicators": ["marginal", "limited impact", "nice to have", "small scale"]
        },
        "Technical Feasibility": {
            "essential_terms": ["feasible", "technology", "platform", "infrastructure", "API"],
            "positive_indicators": ["proven technology", "existing tools", "scalable", "maintainable"],
            "negative_indicators": ["challenging", "complex integration", "legacy system", "constraints"]
        },
        "Timeline Alignment": {
            "essential_terms": ["timeline", "deadline", "schedule", "priority", "time"],
            "positive_indicators": ["reasonable timeline", "adequate time", "phased approach"],
            "negative_indicators": ["tight deadline", "unrealistic", "rushed", "immediate"]
        }
    }
    
    if criteria_name not in criteria_patterns:
        return 3, "No pattern defined"
    
    pattern = criteria_patterns[criteria_name]
    text_lower = text.lower()
    
    # Calculate scoring components
    essential_score = 0
    positive_score = 0
    negative_score = 0
    
    # Count essential terms
    essential_matches = [term for term in pattern["essential_terms"] if term in text_lower]
    essential_score = min(len(essential_matches) / len(pattern["essential_terms"]) * 2, 2)
    
    # Count positive indicators  
    positive_matches = [term for term in pattern["positive_indicators"] if term in text_lower]
    positive_score = min(len(positive_matches) * 0.5, 1.5)
    
    # Count negative indicators
    negative_matches = [term for term in pattern["negative_indicators"] if term in text_lower]
    negative_score = min(len(negative_matches) * 0.4, 1.2)
    
    # Sentiment analysis boost
    sentiment = nlp_analyzer.analyze_sentiment(text)
    sentiment_boost = 0
    if sentiment:
        if sentiment['label'] == 'POSITIVE' and sentiment['confidence'] > 0.7:
            sentiment_boost = 0.3
        elif sentiment['label'] == 'NEGATIVE' and sentiment['confidence'] > 0.7:
            sentiment_boost = -0.3
    
    # TF-IDF keyword relevance
    keywords = nlp_analyzer.extract_keywords(text, top_k=15)
    relevant_keywords = [kw[0] for kw in keywords if any(term in kw[0] for term in pattern["essential_terms"])]
    keyword_boost = min(len(relevant_keywords) * 0.1, 0.5)
    
    # Calculate final score
    base_score = 2 + essential_score + positive_score - negative_score + sentiment_boost + keyword_boost
    final_score = max(1, min(5, round(base_score)))
    
    # Generate detailed reasoning
    reasoning_parts = []
    if essential_matches:
        reasoning_parts.append(f"✓ Key terms: {', '.join(essential_matches[:3])}")
    if positive_matches:
        reasoning_parts.append(f"✓ Positive: {', '.join(positive_matches[:2])}")
    if negative_matches:
        reasoning_parts.append(f"⚠ Concerns: {', '.join(negative_matches[:2])}")
    if relevant_keywords:
        reasoning_parts.append(f"🔍 TF-IDF keywords: {', '.join(relevant_keywords[:2])}")
    if sentiment and sentiment['confidence'] > 0.6:
        reasoning_parts.append(f"😊 Sentiment: {sentiment['label']} ({sentiment['confidence']:.2f})")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Limited relevant content found"
    
    return final_score, reasoning

def get_project_type(text):
    """Enhanced project type detection"""
    if not text:
        return "Unknown"
    
    text_lower = text.lower()
    project_indicators = {
        'Machine Learning': ['ml', 'ai', 'prediction', 'classification', 'regression', 'neural network', 'deep learning'],
        'Web Application': ['web', 'website', 'webapp', 'frontend', 'backend', 'api', 'rest'],
        'Mobile Application': ['mobile', 'app', 'ios', 'android', 'smartphone', 'tablet'],
        'Data Analytics': ['data', 'analytics', 'dashboard', 'reporting', 'visualization', 'business intelligence'],
        'Automation': ['automate', 'workflow', 'process', 'efficiency', 'streamline', 'optimize'],
        'E-commerce': ['shop', 'store', 'payment', 'cart', 'checkout', 'inventory'],
        'Cloud Platform': ['cloud', 'aws', 'azure', 'gcp', 'serverless', 'microservices']
    }
    
    scores = {}
    for project_type, indicators in project_indicators.items():
        score = sum(1 for indicator in indicators if indicator in text_lower)
        scores[project_type] = score
    
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "General Software"

# Header
st.title("🎯 AI Fit Scoring Dashboard")
st.markdown("**Phase 3: With Advanced NLP Analysis** 🧠")
st.write("Testing scikit-learn TF-IDF, TextBlob sentiment analysis, and smart keyword extraction")

# Sidebar for input method
st.sidebar.title("📋 Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["📄 Upload Document", "✍️ Manual Scoring"],
    key="input_method"
)

# Initialize NLP analyzer
nlp_analyzer = NLPAnalyzer()

# Document Upload Section
if input_method == "📄 Upload Document":
    st.subheader("📤 Document Upload & Advanced NLP Analysis")
    
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
            
            # Advanced NLP Analysis
            with st.expander("🧠 Advanced NLP Analysis Results"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # TF-IDF Keywords
                    if SKLEARN_AVAILABLE:
                        keywords = nlp_analyzer.extract_keywords(extracted_text, top_k=10)
                        if keywords:
                            st.subheader("🔍 Top Keywords (TF-IDF)")
                            for keyword, score in keywords:
                                st.write(f"• **{keyword}** ({score:.3f})")
                    
                    # Project insights
                    insights = nlp_analyzer.extract_project_insights(extracted_text)
                    if insights.get('technologies'):
                        st.subheader("💻 Detected Technologies")
                        for tech in insights['technologies']:
                            st.write(f"• {tech}")
                
                with col2:
                    # Sentiment Analysis
                    if TEXTBLOB_AVAILABLE:
                        sentiment = nlp_analyzer.analyze_sentiment(extracted_text)
                        if sentiment:
                            st.subheader("😊 Sentiment Analysis")
                            st.write(f"**Label:** {sentiment['label']}")
                            st.write(f"**Confidence:** {sentiment['confidence']:.2f}")
                            st.write(f"**Polarity:** {sentiment['polarity']:.2f} (-1 to 1)")
                            st.write(f"**Subjectivity:** {sentiment['subjectivity']:.2f} (0 to 1)")
                    
                    # Timeline mentions
                    if insights.get('timeline_mentions'):
                        st.subheader("📅 Timeline References")
                        for mention in insights['timeline_mentions']:
                            if isinstance(mention, tuple):
                                st.write(f"• {' '.join(mention)}")
                            else:
                                st.write(f"• {mention}")
            
            # Document preview
            with st.expander("📖 Document Preview"):
                preview_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                st.text_area("Content Preview:", preview_text, height=200, disabled=True, key="preview")
            
            # Auto-scoring buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧠 Advanced NLP Scoring", type="primary", key="nlp_score"):
                    with st.spinner("🔍 Running advanced NLP analysis..."):
                        for criteria_name in criteria.keys():
                            score, reasoning = advanced_keyword_analysis(extracted_text, criteria_name, nlp_analyzer)
                            st.session_state.scores[criteria_name] = score
                            st.session_state.reasoning[criteria_name] = reasoning
                        
                        # Store NLP analysis
                        st.session_state.nlp_analysis = {
                            'project_type': project_type,
                            'word_count': word_count,
                            'keywords': keywords if SKLEARN_AVAILABLE else [],
                            'sentiment': sentiment if TEXTBLOB_AVAILABLE else None,
                            'insights': insights
                        }
                    
                    st.success("🎯 Advanced NLP scoring completed!")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset Scores", key="reset"):
                    st.session_state.scores = {}
                    st.session_state.reasoning = {}
                    st.session_state.nlp_analysis = {}
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
    
    # NLP Insights Panel
    if st.session_state.nlp_analysis:
        st.subheader("🧠 NLP Analysis Summary")
        nlp_data = st.session_state.nlp_analysis
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if nlp_data.get('sentiment'):
                st.metric("Sentiment", nlp_data['sentiment']['label'])
        with col2:
            if nlp_data.get('keywords'):
                st.metric("Key Terms Found", len(nlp_data['keywords']))
        with col3:
            if nlp_data.get('insights', {}).get('technologies'):
                st.metric("Technologies", len(nlp_data['insights']['technologies']))
        with col4:
            st.metric("Project Type", nlp_data.get('project_type', 'Unknown'))
    
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
        
        # Add NLP analysis summary
        if st.session_state.nlp_analysis:
            nlp_data = st.session_state.nlp_analysis
            summary_row = {
                'Criteria': 'NLP_ANALYSIS_SUMMARY',
                'Score (1-5)': 'N/A',
                'Percentage': 'N/A',
                'Weight': 'N/A', 
                'Weighted Score': 'N/A',
                'Reasoning': f"Project Type: {nlp_data.get('project_type', 'Unknown')}; Keywords: {len(nlp_data.get('keywords', []))}; Sentiment: {nlp_data.get('sentiment', {}).get('label', 'Unknown') if nlp_data.get('sentiment') else 'Unknown'}",
                'Description': 'Advanced NLP Analysis Summary'
            }
            report_data.append(summary_row)
        
        if report_data:
            df = pd.DataFrame(report_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Enhanced CSV Report",
                data=csv,
                file_name=f"ai_fit_nlp_assessment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# Enhanced Status indicators
st.markdown("---")
st.subheader("🔧 Advanced System Status")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write(f"📄 **PDF Processing:** {'✅' if PDF_AVAILABLE else '❌'}")
    st.write(f"📄 **DOCX Processing:** {'✅' if DOCX_AVAILABLE else '❌'}")
with col2:
    st.write(f"🔍 **TF-IDF Analysis:** {'✅' if SKLEARN_AVAILABLE else '❌'}")
    st.write(f"😊 **Sentiment Analysis:** {'✅' if TEXTBLOB_AVAILABLE else '❌'}")
with col3:
    st.write(f"📊 **Visualization:** ✅")
    st.write(f"🧠 **NLP Processing:** {'✅' if SKLEARN_AVAILABLE and TEXTBLOB_AVAILABLE else '❌'}")
with col4:
    st.write(f"📈 **Advanced Analytics:** {'✅' if SKLEARN_AVAILABLE else '❌'}")
    st.write(f"🔤 **Text Processing:** {'✅' if NLTK_AVAILABLE else '❌'}")

st.markdown("**✅ Phase 3 Complete:** Advanced NLP analysis with TF-IDF, sentiment analysis, and smart insights!")
st.markdown("**🎯 Success:** Your AI Fit Scoring Tool now has enterprise-grade NLP capabilities!")