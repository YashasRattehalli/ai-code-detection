import logging
import time
from typing import Any, Dict, Optional

import streamlit as st

# Import the inference pipeline
from ai_code_detector.inference_pipeline import InferencePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Code Detective 🕵️‍♂️",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stChat > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd !important;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5 !important;
        border-left: 4px solid #9c27b0;
    }
    .code-block {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .ai-generated {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .human-written {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name: str) -> Optional[InferencePipeline]:
    """Load and cache the inference pipeline."""
    try:
        with st.spinner(f"Loading {model_name} model..."):
            pipeline = InferencePipeline(model_name=model_name, threshold=0.5)
        st.success(f"✅ {model_name.title()} model loaded successfully!")
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to load {model_name} model: {str(e)}")
        logger.error(f"Model loading error: {e}")
        return None

def analyze_code(code: str, pipeline: InferencePipeline) -> Dict[str, Any]:
    """Analyze code using the inference pipeline."""
    try:
        result = pipeline.predict_single(code)
        return result
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": str(e)}

def display_analysis_result(result: Dict[str, Any], code: str):
    """Display the analysis result in a formatted way."""
    if "error" in result:
        st.error(f"❌ Analysis failed: {result['error']}")
        return

    probability = result.get('probability', 0)
    is_ai_generated = result.get('is_ai_generated', False)
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main prediction
        if is_ai_generated:
            st.markdown(f"""
            <div class="prediction-box ai-generated">
                <h3>🤖 AI-Generated Code Detected</h3>
                <p><strong>Confidence:</strong> {probability:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box human-written">
                <h3>👨‍💻 Human-Written Code Detected</h3>
                <p><strong>Confidence:</strong> {(1-probability):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Confidence visualization
        st.subheader("Confidence Meter")
        
        # Create a progress bar
        if is_ai_generated:
            st.metric("AI Probability", f"{probability:.1%}")
            st.progress(probability)
        else:
            st.metric("Human Probability", f"{(1-probability):.1%}")
            st.progress(1-probability)
        
        # Additional metrics
        st.metric("Code Length", f"{len(code)} chars")
        st.metric("Lines", f"{code.count(chr(10)) + 1}")

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🕵️‍♂️ AI Code Detective")
    st.markdown("### Detect whether code was written by AI or humans")
    st.markdown("---")
    
    # Sidebar for model selection and information
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Model selection
        model_options = {
            "XGBoost": "xgboost",
            "UniXcoder": "embedding_classifier"
        }
        
        selected_model_display = st.selectbox(
            "Select Detection Model",
            options=list(model_options.keys()),
            index=0,
            help="Choose the AI model for code detection"
        )
        
        selected_model = model_options[selected_model_display]
        
        # Load the selected model
        pipeline = load_model(selected_model)
        
        st.markdown("---")
        
        # Model information
        st.header("📊 Model Info")
        if selected_model == "xgboost":
            st.info("""
            **XGBoost Model**
            - Fast tree-based classifier
            - Good for general code detection
            - Lightweight and efficient
            """)
        else:
            st.info("""
            **UniXcoder Model**
            - Transformer-based embeddings
            - Deep understanding of code semantics
            - More accurate but slower
            """)
        
        st.markdown("---")
        
        # Instructions
        st.header("💡 How to Use")
        st.markdown("""
        1. **Paste your code** in the text area
        2. **Click 'Analyze Code'** button
        3. **View the results** with confidence scores
        4. **Try different models** for comparison
        """)
        
        st.markdown("---")
        
        # Examples
        with st.expander("📝 Example Code Snippets"):
            st.code("""
# Python function
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
            """, language="python")
            
            st.code("""
// JavaScript arrow function
const factorial = (n) => {
    return n <= 1 ? 1 : n * factorial(n - 1);
};
            """, language="javascript")
    
    # Main content area
    if pipeline is None:
        st.error("⚠️ Model not loaded. Please check the model files and try again.")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your AI Code Detective 🕵️‍♂️. Paste any code snippet and I'll tell you if it was likely written by AI or a human. What code would you like me to analyze?"
            }
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If this is an analysis result, display the formatted result
            if message["role"] == "assistant" and "analysis_result" in message:
                display_analysis_result(message["analysis_result"], message.get("analyzed_code", ""))
    
    # Chat input
    if prompt := st.chat_input("Paste your code here for analysis..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing code..."):
                # Simulate some processing time for better UX
                time.sleep(0.5)
                
                # Analyze the code
                result = analyze_code(prompt, pipeline)
                
                if "error" not in result:
                    probability = result.get('probability', 0)
                    is_ai_generated = result.get('is_ai_generated', False)
                    
                    if is_ai_generated:
                        response = f"🤖 **Analysis Complete!** This code appears to be **AI-generated** with {probability:.1%} confidence."
                    else:
                        response = f"👨‍💻 **Analysis Complete!** This code appears to be **human-written** with {(1-probability):.1%} confidence."
                    
                    st.markdown(response)
                    
                    # Display detailed results
                    display_analysis_result(result, prompt)
                    
                    # Add assistant message to chat history with analysis result
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "analysis_result": result,
                        "analyzed_code": prompt
                    })
                else:
                    error_response = f"❌ Sorry, I encountered an error while analyzing your code: {result['error']}"
                    st.markdown(error_response)
                    st.session_state.messages.append({"role": "assistant", "content": error_response})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🔬 Powered by AI Code Detection Models | Built with Streamlit</p>
        <p><em>Note: This tool provides probability estimates. Use with discretion for important decisions.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 