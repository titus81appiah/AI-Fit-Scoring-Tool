# 🤖 AI Fit Scoring Tool

A sophisticated Streamlit app that evaluates whether customer problems are suitable for AI solutions using advanced NLP analysis and intelligent scoring metrics.

## 🚀 Live Demo

**Try it now:** [https://ai-fit-scoring-tool.streamlit.app/](https://ai-fit-scoring-tool.streamlit.app/)

## ✨ Features

- **Advanced NLP Analysis**: Powered by spaCy, scikit-learn, and Transformers
- **7 AI-Fit Metrics**: Comprehensive evaluation across multiple dimensions
- **Intelligent Scoring**: Percentage-based scoring with clear verdicts
- **Results Visualization**: Beautiful cards with tags and progress indicators
- **Detailed Analysis**: Identify strengths and areas for improvement
- **Export Reports**: Download comprehensive JSON reports
- **Responsive UI**: Clean interface with improved borders and styling

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository
```bash
git clone https://github.com/titus81appiah/AI-Fit-Scoring-Tool.git
cd AI-Fit-Scoring-Tool
```

2. Create and activate virtual environment
```bash
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
streamlit run app.py
```

## 🔍 How It Works

The tool analyzes customer problem descriptions using advanced NLP techniques:

1. **Text Processing**: Tokenization, stopword removal, and normalization
2. **Feature Extraction**: Identifies AI-relevant keywords and patterns
3. **Multi-Metric Scoring**: Evaluates across 7 key dimensions:
   - Data Availability
   - Pattern Recognition
   - Automation Potential
   - Complexity Level
   - Repetitive Tasks
   - Decision Making
   - Scalability
4. **Intelligent Analysis**: Provides weighted scoring with actionable insights

## 📊 Scoring Metrics

- **Data Availability**: Assesses data quality and accessibility
- **Pattern Recognition**: Identifies recurring patterns and trends
- **Automation Potential**: Evaluates manual vs. automated processes
- **Complexity Level**: Determines if problem complexity warrants AI
- **Repetitive Tasks**: Identifies routine, automatable workflows
- **Decision Making**: Assesses decision complexity and frequency
- **Scalability**: Evaluates growth and volume considerations

## 🎯 Use Cases

- **Product Managers**: Evaluate AI project feasibility
- **Data Scientists**: Assess data requirements and complexity
- **Business Analysts**: Identify automation opportunities
- **Consultants**: Client problem assessment and recommendations
- **Startups**: Validate AI solution viability

## 🚀 Deployment

This app is deployed on Streamlit Cloud for easy access and sharing. The deployed version includes:
- Latest NLP enhancements
- Improved UI with better borders and styling
- Enhanced result cards with tags
- Optimized performance for production use

## 📁 Project Structure

```
AI-Fit-Scoring-Tool/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore rules
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- **Live App**: [https://ai-fit-scoring-tool.streamlit.app/](https://ai-fit-scoring-tool.streamlit.app/)
- **GitHub Repository**: [https://github.com/titus81appiah/AI-Fit-Scoring-Tool](https://github.com/titus81appiah/AI-Fit-Scoring-Tool)
- **Issues**: [https://github.com/titus81appiah/AI-Fit-Scoring-Tool/issues](https://github.com/titus81appiah/AI-Fit-Scoring-Tool/issues)
