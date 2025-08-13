import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Set, Tuple, Dict

import streamlit as st


STOPWORDS: Set[str] = {
	"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
	"as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
	"by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
	"further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
	"himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "let", "me",
	"more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
	"other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
	"some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
	"these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
	"we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
	"you", "your", "yours", "yourself", "yourselves"
}


def tokenize(text: str, min_len: int = 3) -> List[str]:
	lowered = text.lower()
	tokens = re.findall(r"[a-z0-9]+", lowered)
	filtered = [t for t in tokens if len(t) >= min_len and t not in STOPWORDS]
	return filtered


def compute_frequency(tokens: List[str]) -> Dict[str, int]:
	counts: Dict[str, int] = {}
	for token in tokens:
		counts[token] = counts.get(token, 0) + 1
	return counts


def build_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
	if n <= 1:
		return set((t,) for t in tokens)
	ngrams: Set[Tuple[str, ...]] = set()
	for i in range(len(tokens) - n + 1):
		ngrams.add(tuple(tokens[i : i + n]))
	return ngrams


@dataclass
class AIScore:
	score: float
	data_availability: float
	pattern_recognition: float
	automation_potential: float
	complexity_level: float
	repetitive_tasks: float
	decision_making: float
	scalability: float
	verdict: str
	strengths: List[str]
	weaknesses: List[str]
	params: Dict[str, float]


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
	if not a and not b:
		return 0.0
	union = a | b
	if not union:
		return 0.0
	return len(a & b) / len(union)


def overlap_ratio(subset: Set[str], superset: Set[str]) -> float:
	if not superset:
		return 0.0
	return len(subset & superset) / len(superset)


def extract_keywords(tokens: List[str], max_keywords: int) -> List[str]:
	freq = compute_frequency(tokens)
	sorted_items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
	return [k for k, _ in sorted_items[:max_keywords]]


def compute_ai_fit(
	problem_text: str,
	*,
	min_token_len: int = 3,
	keyword_count: int = 20,
) -> AIScore:
	tokens = tokenize(problem_text, min_len=min_token_len)
	token_set = set(tokens)
	
	# AI-fit indicators
	ai_keywords = {
		"data": ["data", "dataset", "database", "csv", "json", "api", "log", "record", "entry"],
		"pattern": ["pattern", "trend", "correlation", "relationship", "sequence", "cycle", "repetition"],
		"automation": ["manual", "repetitive", "routine", "process", "workflow", "step", "task"],
		"complexity": ["complex", "complicated", "multiple", "various", "different", "varied", "diverse"],
		"repetitive": ["repeat", "recurring", "daily", "weekly", "monthly", "regular", "consistent"],
		"decision": ["decision", "choice", "option", "select", "choose", "determine", "evaluate"],
		"scale": ["scale", "grow", "expand", "increase", "volume", "quantity", "amount"]
	}
	
	scores = {}
	strengths = []
	weaknesses = []
	
	# Data availability
	data_tokens = set(ai_keywords["data"])
	data_score = overlap_ratio(token_set, data_tokens) * 100
	scores["data_availability"] = data_score
	if data_score > 50:
		strengths.append("Good data availability indicators")
	else:
		weaknesses.append("Limited data availability mentioned")
	
	# Pattern recognition
	pattern_tokens = set(ai_keywords["pattern"])
	pattern_score = overlap_ratio(token_set, pattern_tokens) * 100
	scores["pattern_recognition"] = pattern_score
	if pattern_score > 30:
		strengths.append("Clear pattern recognition potential")
	else:
		weaknesses.append("Pattern recognition opportunities not obvious")
	
	# Automation potential
	automation_tokens = set(ai_keywords["automation"])
	automation_score = overlap_ratio(token_set, automation_tokens) * 100
	scores["automation_potential"] = automation_score
	if automation_score > 40:
		strengths.append("High automation potential")
	else:
		weaknesses.append("Automation potential unclear")
	
	# Complexity level
	complexity_tokens = set(ai_keywords["complexity"])
	complexity_score = min(100, overlap_ratio(token_set, complexity_tokens) * 200)
	scores["complexity_level"] = complexity_score
	if complexity_score > 60:
		strengths.append("Sufficient complexity for AI value")
	else:
		weaknesses.append("May be too simple for AI")
	
	# Repetitive tasks
	repetitive_tokens = set(ai_keywords["repetitive"])
	repetitive_score = overlap_ratio(token_set, repetitive_tokens) * 100
	scores["repetitive_tasks"] = repetitive_score
	if repetitive_score > 40:
		strengths.append("Repetitive tasks identified")
	else:
		weaknesses.append("Repetitive nature not clear")
	
	# Decision making
	decision_tokens = set(ai_keywords["decision"])
	decision_score = overlap_ratio(token_set, decision_tokens) * 100
	scores["decision_score"] = decision_score
	if decision_score > 30:
		strengths.append("Decision-making elements present")
	else:
		weaknesses.append("Decision-making aspects unclear")
	
	# Scalability
	scale_tokens = set(ai_keywords["scale"])
	scale_score = overlap_ratio(token_set, scale_tokens) * 100
	scores["scalability"] = scale_score
	if scale_score > 30:
		strengths.append("Scalability considerations present")
	else:
		weaknesses.append("Scalability not addressed")
	
	# Overall score (weighted average)
	weights = {
		"data_availability": 0.25,
		"pattern_recognition": 0.20,
		"automation_potential": 0.20,
		"complexity_level": 0.15,
		"repetitive_tasks": 0.10,
		"decision_score": 0.05,
		"scalability": 0.05
	}
	
	total_score = sum(scores[k] * weights[k] for k in weights.keys())
	
	# Verdict
	if total_score >= 80:
		verdict = "Excellent AI fit"
	elif total_score >= 60:
		verdict = "Good AI fit"
	elif total_score >= 40:
		verdict = "Moderate AI fit"
	else:
		verdict = "Poor AI fit"
	
	return AIScore(
		score=total_score,
		data_availability=data_score,
		pattern_recognition=pattern_score,
		automation_potential=automation_score,
		complexity_level=complexity_score,
		repetitive_tasks=repetitive_score,
		decision_making=decision_score,
		scalability=scale_score,
		verdict=verdict,
		strengths=strengths,
		weaknesses=weaknesses,
		params={
			"min_token_len": float(min_token_len),
			"keyword_count": float(keyword_count),
		},
	)


st.set_page_config(page_title="AI Fit Scoring Tool", page_icon="🤖", layout="centered")

st.title("🤖 AI Fit Scoring Tool")
st.write(
	"Evaluate whether a customer problem is suitable for an AI solution using transparent metrics."
)

with st.sidebar:
	st.header("Settings")
	st.caption("Adjust analysis parameters.")
	keyword_count = st.slider("Keywords to extract", 10, 50, 20, 1)
	min_token_len = st.slider("Minimum token length", 2, 6, 3, 1)
	
	sample_clicked = st.button("Load sample problem")

default_problem = (
	"We need to analyze customer support tickets to identify common patterns and automate responses. "
	"Currently, our team manually reviews hundreds of tickets daily, categorizing them and writing "
	"custom responses. We have a database of historical tickets with customer messages, categories, "
	"and resolution times. We want to scale this process as our customer base grows."
)

if sample_clicked:
	st.session_state["problem_text"] = default_problem
	st.rerun()

problem_text = st.text_area(
	"Customer Problem Description",
	value=st.session_state.get("problem_text", ""),
	height=250,
	placeholder=default_problem,
	key="problem_text",
)

score_clicked = st.button("Score AI Fit")

if score_clicked:
	result = compute_ai_fit(
		problem_text=problem_text,
		min_token_len=min_token_len,
		keyword_count=keyword_count,
	)

	col1, col2 = st.columns([1, 2])
	with col1:
		st.metric("AI Fit Score", f"{result.score:.1f}%")
		st.progress(min(100, int(result.score)))
		st.success(result.verdict)
	
	with col2:
		st.write("**Score Breakdown:**")
		st.write(f"• Data Availability: {result.data_availability:.1f}%")
		st.write(f"• Pattern Recognition: {result.pattern_recognition:.1f}%")
		st.write(f"• Automation Potential: {result.automation_potential:.1f}%")
		st.write(f"• Complexity Level: {result.complexity_level:.1f}%")
		st.write(f"• Repetitive Tasks: {result.repetitive_tasks:.1f}%")
		st.write(f"• Decision Making: {result.decision_making:.1f}%")
		st.write(f"• Scalability: {result.scalability:.1f}%")

	st.subheader("Analysis")
	cols = st.columns(2)
	with cols[0]:
		st.caption("Strengths")
		for strength in result.strengths:
			st.write(f"✅ {strength}")
		if not result.strengths:
			st.write("—")
	
	with cols[1]:
		st.caption("Areas for Improvement")
		for weakness in result.weaknesses:
			st.write(f"⚠️ {weakness}")
		if not result.weaknesses:
			st.write("—")

	report = {
		"timestamp": datetime.utcnow().isoformat() + "Z",
		"inputs": {
			"problem_text": problem_text,
		},
		"result": {
			"overall_score": result.score,
			"verdict": result.verdict,
			"data_availability_percent": result.data_availability,
			"pattern_recognition_percent": result.pattern_recognition,
			"automation_potential_percent": result.automation_potential,
			"complexity_level_percent": result.complexity_level,
			"repetitive_tasks_percent": result.repetitive_tasks,
			"decision_making_percent": result.decision_making,
			"scalability_percent": result.scalability,
			"strengths": result.strengths,
			"weaknesses": result.weaknesses,
		},
		"params": result.params,
	}

	st.download_button(
		label="Download JSON report",
		data=json.dumps(report, indent=2),
		file_name="ai_fit_score_report.json",
		mime="application/json",
	)
