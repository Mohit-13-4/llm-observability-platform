"""
LLM Observability Dashboard - Visualize Hallucination Detection Results
Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedder import EmbeddingEngine
from llm.judge import LLMJudge
from modules.hallucination import HallucinationDetector

# Page config
st.set_page_config(
    page_title="LLM Observability Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .hallucination-badge {
        padding: 0.5rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    .contradiction-warning {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .success-banner {
        background-color: #00cc66;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    /* Button styling */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    /* Results container */
    .results-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    /* Metric card styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluator' not in st.session_state:
    with st.spinner("Loading models... This may take a moment..."):
        try:
            embedder = EmbeddingEngine()
            llm_judge = LLMJudge()
            st.session_state.evaluator = HallucinationDetector(embedder, llm_judge)
            st.session_state.models_loaded = True
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.session_state.models_loaded = False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("🔍 LLM Observability")
    st.markdown("---")
    
    st.markdown("### 📊 Evaluation Methods")
    st.markdown("""
    - **Embedding Similarity**: Semantic comparison
    - **Contradiction Detection**: Direct contradictions
    - **LLM-as-Judge**: Second LLM evaluation (phi-2)
    - **Ground Truth**: Optional comparison
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Scoring Weights")
    st.markdown("""
    - **LLM Judge: 80%** (Primary)
    - **Embedding Similarity: 20%** (Secondary)
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Confidence Levels")
    st.markdown("""
    - 🟢 **High**: Score < 0.3
    - 🟡 **Medium**: 0.3-0.5
    - 🟠 **Low**: 0.5-0.7
    - 🔴 **Severe**: > 0.7
    """)
    
    st.markdown("---")
    st.markdown("### 🧪 Quick Test Examples")
    
    if st.button("🌱 Watermelon Myth", use_container_width=True):
        st.session_state.example_question = "What happens if you eat watermelon seeds?"
        st.session_state.example_answer = "A watermelon will grow in your stomach"
        st.session_state.example_context = "Watermelon seeds are harmless. They pass through your digestive system. They will NOT grow into a watermelon."
        st.session_state.example_ground_truth = "Nothing, they pass through your digestive system."
        st.rerun()
    
    if st.button("🏛️ Great Wall Myth", use_container_width=True):
        st.session_state.example_question = "Is the Great Wall of China visible from space?"
        st.session_state.example_answer = "Yes, it's the only man-made structure visible from space"
        st.session_state.example_context = "The Great Wall is not visible from space with the naked eye. This is a common myth. Astronauts cannot see it without aid."
        st.session_state.example_ground_truth = "No, it is not visible from space."
        st.rerun()
    
    if st.button("🇫🇷 Correct Capital", use_container_width=True):
        st.session_state.example_question = "What is the capital of France?"
        st.session_state.example_answer = "Paris"
        st.session_state.example_context = "France is a country in Western Europe. Its capital city is Paris."
        st.session_state.example_ground_truth = "Paris"
        st.rerun()
    
    if st.button("🇮🇳 Wrong Capital (Mumbai)", use_container_width=True):
        st.session_state.example_question = "What is the capital of India?"
        st.session_state.example_answer = "Mumbai is the capital of India"
        st.session_state.example_context = "India is a country in South Asia. Its capital city is New Delhi."
        st.session_state.example_ground_truth = "New Delhi"
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Benchmark Results")
    st.markdown("""
    - **SQuAD Accuracy**: 100%
    - **HotpotQA Accuracy**: 86.7%
    - **TruthfulQA Accuracy**: 86.7%
    - **Precision**: 1.000
    - **F1 Score**: 0.889
    """)

# Main content
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🔍 LLM Observability & Evaluation Platform")
st.markdown("Real-time hallucination detection and quality analysis")
st.markdown("**Powered by phi-2 (1.5B) and Sentence Transformers**")
st.markdown('</div>', unsafe_allow_html=True)

# Check if models are loaded
if not st.session_state.get('models_loaded', False):
    st.error("⚠️ Models failed to load. Please check your installation and restart.")
    st.stop()

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Input")
    
    # Use example if set
    default_question = st.session_state.get('example_question', "What is the capital of France?")
    default_answer = st.session_state.get('example_answer', "The capital of France is Paris.")
    default_context = st.session_state.get('example_context', "France is a country in Western Europe. Its capital is Paris.")
    default_ground_truth = st.session_state.get('example_ground_truth', "Paris")
    
    question = st.text_area(
        "**Question / Prompt**",
        value=default_question,
        height=80,
        help="Enter the question or prompt given to the LLM"
    )
    
    answer = st.text_area(
        "**LLM Answer**",
        value=default_answer,
        height=80,
        help="The answer generated by the LLM to evaluate"
    )
    
    context = st.text_area(
        "**Context (Optional - for RAG)**",
        value=default_context,
        height=80,
        help="Context for RAG scenarios (optional)"
    )
    
    ground_truth = st.text_input(
        "**Ground Truth (Optional)**",
        value=default_ground_truth,
        help="Correct answer for comparison (optional)"
    )

with col2:
    st.markdown("### ⚙️ Settings")
    
    show_confidence = st.checkbox("Show Confidence Score", value=True)
    show_method_details = st.checkbox("Show Method Details", value=True)
    show_llm_explanation = st.checkbox("Show LLM Explanation", value=True)
    show_contradiction_warning = st.checkbox("Show Contradiction Warning", value=True)
    
    st.markdown("---")
    st.markdown("### 🎨 Visualization")
    chart_type = st.selectbox("Chart Type", ["Gauge", "Radial", "Bar", "Donut"])

# Evaluate Button
evaluate_col1, evaluate_col2, evaluate_col3 = st.columns([1, 2, 1])
with evaluate_col2:
    if st.button("🚀 Evaluate", type="primary", use_container_width=True):
        if not question or not answer:
            st.error("Please provide both a question and an answer!")
        else:
            with st.spinner("Analyzing... This may take a few seconds..."):
                # Use context or fallback to question
                context_to_use = context if context else question
                
                # Run evaluation
                result = st.session_state.evaluator.detect(
                    answer=answer,
                    context=context_to_use,
                    ground_truth=ground_truth if ground_truth else None
                )
                
                # Display Results
                st.markdown("---")
                st.markdown("## 📊 Evaluation Results")
                
                # Contradiction Warning
                if show_contradiction_warning and result.get("contradiction_detected", False):
                    st.markdown('<div class="contradiction-warning">⚠️ CONTRADICTION DETECTED! The answer directly contradicts the context.</div>', unsafe_allow_html=True)
                
                # Metrics Row
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    hallucination_score = result["hallucination_score"]
                    if hallucination_score < 0.3:
                        status = "🟢 Low Risk"
                        delta_color = "normal"
                    elif hallucination_score < 0.5:
                        status = "🟡 Medium Risk"
                        delta_color = "off"
                    elif hallucination_score < 0.7:
                        status = "🟠 High Risk"
                        delta_color = "inverse"
                    else:
                        status = "🔴 Severe Risk"
                        delta_color = "inverse"
                    
                    st.metric(
                        label="Hallucination Score",
                        value=f"{hallucination_score:.3f}",
                        delta=status,
                        delta_color=delta_color
                    )
                
                with metric_col2:
                    if show_confidence and "confidence" in result:
                        confidence = result['confidence']
                        if confidence > 0.7:
                            conf_status = "High"
                        elif confidence > 0.5:
                            conf_status = "Medium"
                        else:
                            conf_status = "Low"
                        st.metric(
                            label="Confidence",
                            value=f"{confidence:.1%}",
                            delta=conf_status
                        )
                    else:
                        st.metric(label="Verdict", value=result["verdict"][:40])
                
                with metric_col3:
                    st.metric(
                        label="Methods Used",
                        value=len(result["methods"]),
                        delta="Detection Methods"
                    )
                
                with metric_col4:
                    st.metric(
                        label="Status",
                        value="✅ Evaluated",
                        delta=datetime.now().strftime("%H:%M:%S")
                    )
                
                # Gauge Chart
                st.markdown("### 🎯 Hallucination Score Visualization")
                
                if chart_type == "Gauge":
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=hallucination_score * 100,
                        title={'text': "Hallucination Risk (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if hallucination_score > 0.7 else "orange" if hallucination_score > 0.3 else "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightsalmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': hallucination_score * 100
                            }
                        }
                    ))
                    fig.update_layout(height=400, margin=dict(l=50, r=50, t=80, b=50))
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Radial":
                    fig = go.Figure(go.Barpolar(
                        r=[hallucination_score * 100, 100 - hallucination_score * 100],
                        theta=['Hallucination', 'Faithful'],
                        width=[0.5, 0.5],
                        marker_color=['#ff4b4b', '#00cc66'],
                        opacity=0.8
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Donut":
                    fig = go.Figure(data=[go.Pie(
                        labels=['Hallucination', 'Faithful'],
                        values=[hallucination_score * 100, (1 - hallucination_score) * 100],
                        hole=0.4,
                        marker_colors=['#ff4b4b', '#00cc66'],
                        textinfo='percent+label'
                    )])
                    fig.update_layout(title="Hallucination vs Faithful", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # Bar
                    fig = go.Figure(go.Bar(
                        x=['Hallucination Score', 'Faithfulness Score'],
                        y=[hallucination_score * 100, (1 - hallucination_score) * 100],
                        marker_color=['#ff4b4b', '#00cc66'],
                        text=[f"{hallucination_score*100:.1f}%", f"{(1-hallucination_score)*100:.1f}%"],
                        textposition='auto',
                        textfont=dict(size=14, color='white')
                    ))
                    fig.update_layout(
                        title="Hallucination vs Faithfulness",
                        yaxis_title="Percentage",
                        yaxis=dict(range=[0, 100]),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Method Details
                if show_method_details:
                    st.markdown("### 🔬 Method Breakdown")
                    
                    methods_data = []
                    for method_name, method_result in result["methods"].items():
                        if isinstance(method_result, dict):
                            if method_name == "llm_judge":
                                methods_data.append({
                                    "Method": "🤖 LLM-as-Judge (phi-2)",
                                    "Result": "✅ Supported" if method_result.get("is_supported") else "❌ Not Supported",
                                    "Details": method_result.get("explanation", "")[:150]
                                })
                            elif method_name == "context_similarity":
                                methods_data.append({
                                    "Method": "📊 Embedding Similarity",
                                    "Result": f"{method_result:.3f}",
                                    "Details": "Semantic similarity between answer and context"
                                })
                            else:
                                methods_data.append({
                                    "Method": method_name.replace("_", " ").title(),
                                    "Result": str(method_result).replace("_", " ")[:50],
                                    "Details": ""
                                })
                        else:
                            methods_data.append({
                                "Method": method_name.replace("_", " ").title(),
                                "Result": str(method_result)[:50],
                                "Details": ""
                            })
                    
                    df = pd.DataFrame(methods_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                # LLM Explanation
                if show_llm_explanation and "llm_judge" in result["methods"]:
                    st.markdown("### 🤖 LLM Judge Explanation")
                    llm_result = result["methods"]["llm_judge"]
                    bg_color = "#e8f5e9" if llm_result.get("is_supported") else "#ffebee"
                    st.markdown(f'<div style="background-color: {bg_color}; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">', unsafe_allow_html=True)
                    st.markdown(f"**LLM Judge Verdict:** {'✅ Supported' if llm_result.get('is_supported') else '❌ Not Supported'}")
                    st.markdown(f"**Explanation:** {llm_result.get('explanation', 'No explanation provided')[:500]}")
                    st.markdown(f"**Confidence:** {llm_result.get('confidence', 0.5):.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional Metrics
                if "context_similarity" in result["methods"]:
                    st.markdown("### 📈 Similarity Metrics")
                    sim_col1, sim_col2 = st.columns(2)
                    with sim_col1:
                        sim_value = result["methods"]["context_similarity"]
                        sim_color = "green" if sim_value > 0.7 else "orange" if sim_value > 0.4 else "red"
                        st.metric(
                            "Context Similarity",
                            f"{sim_value:.3f}",
                            "Answer vs Context (higher = more faithful)",
                            delta_color="normal"
                        )
                    if "ground_truth_similarity" in result["methods"]:
                        with sim_col2:
                            gt_sim = result["methods"]["ground_truth_similarity"]
                            st.metric(
                                "Ground Truth Similarity",
                                f"{gt_sim:.3f}",
                                "Answer vs Truth (higher = more accurate)"
                            )
                
                # Verdict Banner
                if hallucination_score > 0.7:
                    st.error(f"⚠️ {result['verdict']}")
                elif hallucination_score > 0.4:
                    st.warning(f"⚠️ {result['verdict']}")
                else:
                    st.success(f"✅ {result['verdict']}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p><strong>LLM Observability Platform</strong> | Built with Streamlit, PyTorch, and 🤗 Transformers</p>
    <p>Powered by <strong>Phi-2 (1.5B)</strong> and <strong>Sentence Transformers</strong> | 4-Method Hallucination Detection</p>
    <p><strong>Benchmarks:</strong> SQuAD: 100% | HotpotQA: 86.7% | TruthfulQA: 86.7% | Precision: 1.000 | F1: 0.889</p>
</div>
""", unsafe_allow_html=True)