"""Main streamlit application for Used Car Price Prediction."""

# fmt: off
# isort: skip_file
import base64
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import streamlit as st
from app_components.data_viewer import DataViewer
from app_components.prediction_interface import PredictionInterface
from config.config import get_settings
# fmt: on

# Page configuration
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_base64_of_image(image_path):
    """Convert image to base64 for CSS background."""
    try:
        path = Path(image_path)
        if not path.exists():
            st.warning(f"Image not found at: {path}")
            return None
        with path.open("rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def set_background_image(use_image=True):
    """Set background image with overlay."""
    settings = get_settings()
    base64_image = None

    if use_image:
        image_path = settings.streamlit_image_path / "car_image.png"
        base64_image = get_base64_of_image(image_path=image_path)

    if base64_image:
        st.markdown(
            f"""
        <style>
        /* Remove default Streamlit padding and margins */
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 0rem;
            max-width: 100%;
        }}

        /* Full page background image */
        .stApp {{
            background:
                linear-gradient(
                    rgba(0, 0, 0, 0.65),
                    rgba(0, 0, 0, 0.65)
                ),
                url("data:image/png;base64,{base64_image}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}

        /* Remove top whitespace */
        header {{
            background-color: transparent !important;
        }}

        .main > div:first-child {{
            padding-top: 0 !important;
        }}

        /* Ensure content doesn't have extra spacing */
        section[data-testid="stSidebar"] > div:first-child {{
            padding-top: 2rem;
        }}
    </style>
    """,
            unsafe_allow_html=True,
        )
        return

    # Fallback to dark gradient
    st.markdown(
        """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%) !important;
        background-attachment: fixed !important;
    }

    .main .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Custom CSS for styling
st.markdown(
    """
<style>
    /* Remove all default padding and margins */
    .main > div {
        padding-top: 0rem !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        color: #f7d77e;
        text-shadow: 
            0 0 20px rgba(247, 215, 126, 0.8),
            0 0 40px rgba(247, 215, 126, 0.5),
            2px 2px 10px rgba(0, 0, 0, 0.9);
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }

    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.9);
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* All text elements */
    .stMarkdown:not(h1), p, span, div:not(:has(h1)), label, h2, h3, h4, h5 {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
    }

    h1 {
        color: inherit !important;
    }

    .main-header,
    div[style*="text-align: center"] h1,
    h1[style*="#f7d77e"] {
        color: #f7d77e !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 10, 0.92) !important;
        backdrop-filter: blur(10px);
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #f7d77e !important;
    }

    /* Info boxes */
    .info-box {
        background-color: rgba(0, 20, 20, 0.85);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid rgba(0, 217, 255, 0.3);
        margin-bottom: 1rem;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }

    .info-box h3, .info-box h4 {
        color: #00d9ff !important;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
    }
    
    .info-box ul li {
        color: #e0e0e0 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, 
            rgba(30, 30, 30, 0.9) 0%, 
            rgba(100, 100, 100, 0.9) 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 
            0 10px 40px rgba(161, 164, 165, 0.4),
            inset 0 0 20px rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    .prediction-box h1, .prediction-box h2, .prediction-box p {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }

    .prediction-box h1 {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        margin: 1rem 0 !important;
    }

    /* Metric boxes */
    .metric-box {
        background-color: rgba(30, 30, 30, 0.9);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 217, 255, 0.3);
        text-align: center;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
    }

    /* Streamlit metric widget */
    [data-testid="stMetricValue"] {
        color: #f7d77e !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(247, 215, 126, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: #f7d77e !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
    }

    /* Styled metrics tables */
    .metrics-table {
        background-color: #2a2a2a !important;
        border: 1px solid #f7d77e !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    .metrics-table table {
        width: 100% !important;
        border-collapse: collapse !important;
    }

    .metrics-table th {
        background-color: #2a2a2a !important;
        color: #f7d77e !important;
        padding: 0.75rem !important;
        text-align: left !important;
        font-weight: 700 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    .metrics-table td {
        background-color: #2a2a2a !important;
        color: white !important;
        padding: 0.75rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .metrics-table tr:last-child td {
        border-bottom: none !important;
    }

    .metrics-table tr:hover td {
        background-color: #3a3a3a !important;
    }

    /* SHAP and Importance tables */
    .shap-table,
    .importance-table {
        background-color: #2a2a2a !important;
        border: 1px solid #f7d77e !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    .shap-table table,
    .importance-table table {
        width: 100% !important;
        border-collapse: collapse !important;
    }

    .shap-table th,
    .importance-table th {
        background-color: #2a2a2a !important;
        color: #f7d77e !important;
        padding: 0.75rem !important;
        text-align: left !important;
        font-weight: 700 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    .shap-table td,
    .importance-table td {
        background-color: #2a2a2a !important;
        color: white !important;
        padding: 0.75rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .shap-table tr:last-child td,
    .importance-table tr:last-child td {
        border-bottom: none !important;
    }

    .shap-table tr:hover td,
    .importance-table tr:hover td {
        background-color: #3a3a3a !important;
    }

    /* Warning and error boxes */
    .warning-box {
        background-color: rgba(255, 193, 7, 0.25);
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    .error-box {
        background-color: rgba(220, 53, 69, 0.25);
        border: 2px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }

    /* Tables */
    .dataframe {
        background-color: rgba(30, 30, 30, 0.7) !important;
        color: white !important;
        backdrop-filter: blur(5px);
    }
    
    .dataframe th {
        background-color: rgba(247, 215, 126, 0.2) !important;
        color: #f7d77e !important;
        border: 1px solid #f7d77e !important;
        font-weight: 700 !important;
        text-shadow: 0 0 5px rgba(247, 215, 126, 0.5);
    }
    
    .dataframe td {
        background-color: rgba(30, 30, 30, 0.7) !important;
        color: white !important;
        border: 1px solid #f7d77e !important;
    }

    /* Streamlit dataframe styling */
    [data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] > div,
    div[data-testid="stDataFrame"] > div > div,
    .dataframe-container {
        background-color: rgba(30, 30, 30, 0.7) !important;
        border: 1px solid #f7d77e !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stDataFrame"] div {
        background-color: rgba(30, 30, 30, 0.7) !important;
    }

    /* Streamlit table styling */
    [data-testid="stTable"],
    div[data-testid="stTable"] > div,
    div[data-testid="stTable"] > div > div,
    .table-container {
        background-color: rgba(30, 30, 30, 0.7) !important;
        border: 1px solid #f7d77e !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stTable"] div {
        background-color: rgba(30, 30, 30, 0.7) !important;
    }

    /* Table container */
    .stDataFrame {
        border: 1px solid #f7d77e !important;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Dataframe wrapper */
    .element-container:has([data-testid="stDataFrame"]) {
        background-color: rgba(30, 30, 30, 0.7) !important;
    }

    /* Table cells */
    table.dataframe {
        background-color: rgba(30, 30, 30, 0.7) !important;
    }

    table.dataframe thead tr th {
        background-color: rgba(247, 215, 126, 0.2) !important;
        color: #f7d77e !important;
        border: 1px solid #f7d77e !important;
        font-weight: 700 !important;
    }

    table.dataframe tbody tr td {
        background-color: rgba(30, 30, 30, 0.7) !important;
        color: white !important;
        border: 1px solid #f7d77e !important;
    }

    /* Even/odd row styling */
    table.dataframe tbody tr:nth-child(even) {
        background-color: rgba(0, 0, 0, 0.2) !important;
    }

    table.dataframe tbody tr:nth-child(odd) {
        background-color: rgba(0, 0, 0, 0.1) !important;
    }

    /* Index column */
    table.dataframe tbody tr th {
        background-color: rgba(247, 215, 126, 0.15) !important;
        color: #f7d77e !important;
        border: 1px solid #f7d77e !important;
    }

    /* Input widgets */
    .stTextInput input, .stNumberInput input {
        background-color: #2a2a2a !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #f7d77e !important;
        box-shadow: 0 0 10px rgba(247, 215, 126, 0.3) !important;
        outline: none !important;
    }

    /* Number input container - remove white border */
    .stNumberInput > div > div {
        border: none !important;
        background-color: transparent !important;
    }

    .stNumberInput > div {
        border: none !important;
    }

    /* Remove white borders/edges from number input containers */
    .stNumberInput > div:first-child {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    /* Target the inner wrapper */
    .stNumberInput > div > div:first-child {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    /* Target the input wrapper specifically */
    div[data-baseweb="input"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    /* The input group container */
    div[data-baseweb="base-input"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    /* Remove focus ring that might appear white */
    .stNumberInput *:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Ensure the main input container has no border */
    [data-testid="stNumberInput"] div[data-baseweb="input"] > div {
        border: none !important;
        background: transparent !important;
    }

    /* Style only the actual input field */
    .stNumberInput input[type="number"] {
        background-color: #2a2a2a !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 8px 0 0 8px !important;
    }

    /* Style the button container */
    .stNumberInput div[data-baseweb="input"] > div:last-child {
        background-color: transparent !important;
        border: none !important;
    }

    /* Individual + - buttons */
    .stNumberInput button {
        background-color: #2a2a2a !important;
        color: #f7d77e !important;
        border: 1px solid #444 !important;
        margin-left: 0.25rem !important;
    }

    /* Top button (decrement) */
    .stNumberInput button:first-of-type {
        border-radius: 0 0 0 0 !important;
        border: 1px solid #444 !important
    }

    /* Bottom button (increment) */
    .stNumberInput button:last-of-type {
        border-radius: 0 8px 8px 0 !important;
        border: 1px solid #444 !important;
    }

    /* Remove any white corners/edges */
    .stNumberInput *:not(input):not(button) {
        background: transparent !important;
        border: none !important;
    }

    /* Selectbox (dropdown) styling - dark background */
    .stSelectbox select {
        background-color: #2a2a2a !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }

    .stSelectbox select:focus {
        border-color: #f7d77e !important;
        box-shadow: 0 0 10px rgba(247, 215, 126, 0.3) !important;
        outline: none !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #00d9ff 0%, #0096ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #00ffff 0%, #00b8ff 100%) !important;
        box-shadow: 0 8px 30px rgba(0, 217, 255, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* Dropdown container - remove extra borders */
    .stSelectbox > div > div {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* Dropdown menu options - dark background with white text */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2a2a2a !important;
        color: white !important;
        border: 1px solid #444 !important;
    }

    /* Dropdown list items */
    .stSelectbox [role="listbox"] {
        background-color: #2a2a2a !important;
        border: 1px solid #f7d77e !important;
    }

    .stSelectbox [role="option"] {
        background-color: #2a2a2a !important;
        color: white !important;
    }

    .stSelectbox [role="option"]:hover {
        background-color: #3a3a3a !important;
        color: #f7d77e !important;
    }

    /* Help tooltip - dark background with gold/white text */
    .stTooltipIcon {
        color: #f7d77e !important;
    }

    /* Tooltip popup styling */
    [data-testid="stTooltipHoverTarget"] {
        color: #f7d77e !important;
    }

    /* The actual tooltip content */
    .stTooltipContent,
    [data-testid="stTooltipContent"],
    div[role="tooltip"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #f7d77e !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }

    /* Input labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Number input buttons (+ and -) */
    .stNumberInput button {
        background-color: #2a2a2a !important;
        color: #f7d77e !important;
        border: none !important;
        border-radius: 4px !important;
    }

    .stNumberInput button:hover {
        background-color: #3a3a3a !important;
        color: #f7d77e !important;
    }

    /* Force remove all white borders from input containers */
    [data-testid="stNumberInput"] > div,
    [data-testid="stTextInput"] > div,
    [data-testid="stSelectbox"] > div {
        border: none !important;
    }

    /* Remove inner container borders */
    [data-testid="stNumberInput"] > div > div > div,
    [data-testid="stTextInput"] > div > div,
    [data-testid="stSelectbox"] > div > div {
        border: none !important;
        box-shadow: none !important;
    }

    /* Buttons - gold gradient */
    .stButton button,
    [data-testid="stForm"] button[kind="primary"],
    button[data-testid="stFormSubmitButton"] {
        background: linear-gradient(135deg, #f7d77e 0%, #d4af37 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 5px 20px rgba(247, 215, 126, 0.4) !important;
        transition: all 0.3s ease !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
    }

    .stButton button:hover,
    [data-testid="stForm"] button[kind="primary"]:hover,
    button[data-testid="stFormSubmitButton"]:hover {
        background: linear-gradient(135deg, #ffd700 0%, #f7d77e 100%) !important;
        box-shadow: 0 8px 30px rgba(247, 215, 126, 0.6) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }

    /* Ensure button text is white */
    [data-testid="stForm"] button p,
    button[data-testid="stFormSubmitButton"] p {
        color: #ffffff !important;
    }

    /* More specific for Car Specifications header outside form */
    .stMarkdown h2 {
        color: #f7d77e !important;
    }

    /* Analysis & Insights title - gold without shadow */
    .stMarkdown h3 {
        color: #f7d77e !important;
        text-shadow: none !important;
    }

    /* Ensure h2 headers are gold */
    h2 {
        color: #f7d77e !important;
        text-shadow: 0 0 10px rgba(247, 215, 126, 0.5) !important;
    }

    /* More specific dropdown styling */
    div[data-baseweb="select"] {
        background-color: #2a2a2a !important;
    }


    div[data-baseweb="select"] > div {
        background-color: #2a2a2a !important;
        border-color: #444 !important;
    }

    /* Dropdown arrow */
    div[data-baseweb="select"] svg {
        color: #f7d77e !important;
    }

    /* Selected value in dropdown */
    div[data-baseweb="select"] span {
        color: white !important;
    }

    /* Dropdown menu when opened */
    ul[role="listbox"] {
        background-color: #2a2a2a !important;
        border: 1px solid #f7d77e !important;
    }

    /* Individual dropdown items */
    li[role="option"] {
        background-color: #2a2a2a !important;
        color: white !important;
    }

    li[role="option"]:hover {
        background-color: #3a3a3a !important;
        color: #f7d77e !important;
    }

    /* Individual select items */
    li[role="select"] {
        background-color: #2a2a2a !important;
        color: white !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(20, 20, 20, 0.8);
        border-bottom: 2px solid #f7d77e;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #b3b3b3 !important;
        background-color: transparent;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #f7d77e !important;
        border-bottom: 3px solid #f7d77e !important;
        background-color: rgba(247, 215, 126, 0.1) !important;
        box-shadow: 0 2px 0 0 #dc3545 !important;
        text-shadow: none !important;
    }

    /* Sidebar info */
    .sidebar-info {
        background-color: rgba(30, 30, 30, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #f7d77e;
        backdrop-filter: blur(10px);
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.5);
    }

    .sidebar-info h3, .sidebar-info h4 {
        color: #f7d77e !important;
        text-shadow: 0 0 10px rgba(247, 215, 126, 0.5);
    }
    
    .sidebar-info ul li {
        color: #e0e0e0 !important;
        line-height: 1.8;
    }

    /* Charts */
    .js-plotly-plot {
        border: 1px solid #f7d77e !important;
        border-radius: 10px !important;
        padding: 0.5rem;
    }

    /* Plotly chart container */
    .stPlotlyChart {
        border: 1px solid #f7d77e;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(30, 30, 30, 0.8) !important;
        color: #f7d77e !important;
        border: 2px solid #f7d77e !important;
        border-radius: 8px !important;
        backdrop-filter: blur(10px);
        font-weight: 600 !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(40, 40, 40, 0.9) !important;
        border-color: #f7d77e !important;
        box-shadow: 0 0 10px rgba(247, 215, 126, 0.3);
    }

    .streamlit-expanderContent {
        background-color: rgba(20, 20, 20, 0.7) !important;
        border: 2px solid #f7d77e !important;
        border-top: none !important;
        backdrop-filter: blur(10px);
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }

    /* Style text inside expander */
    .streamlit-expanderContent h4,
    .streamlit-expanderContent .stMarkdown h4,
    div[data-testid="stExpander"] h4 {
        color: #f7d77e !important;
        text-shadow: 0 0 5px rgba(247, 215, 126, 0.3) !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    .streamlit-expanderContent h3,
    .streamlit-expanderContent h5 {
        color: #f7d77e !important;
    }

    .streamlit-expanderContent p,
    .streamlit-expanderContent li {
        color: #e0e0e0 !important;
        line-height: 1.8;
    }

    .streamlit-expanderContent ul {
        margin-left: 1rem;
    }

    .streamlit-expanderContent strong {
        color: #f7d77e !important;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess {
        background-color: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
    }

    .stSuccess > div {
        background-color: transparent !important;
        border: none !important;
    }

    .stSuccess p {
        color: #5dff5d !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8) !important;
    }

    .stSuccess svg {
        color: #5dff5d !important;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.3) !important;
        color: #ffd966 !important;
        border: 1px solid #ffc107 !important;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.3) !important;
        color: #ff6b6b !important;
        border: 1px solid #dc3545 !important;
        backdrop-filter: blur(10px);
    }
    
    .stInfo {
        background-color: rgba(0, 217, 255, 0.2) !important;
        color: #66d9ff !important;
        border: 1px solid #00d9ff !important;
        backdrop-filter: blur(10px);
    }

    /* Form styling */
    [data-testid="stForm"] {
        background-color: transparent !important;
        border: 2px solid #f7d77e !important;
        border-radius: 15px;
        padding: 1.5rem;
        backdrop-filter: none !important;
        box-shadow: 0 0px 20px rgba(247, 215, 126, 0.3);
    }

    /* Form section headers inside columns */
    [data-testid="stForm"] h4 {
        color: #f7d77e !important;
    }

    /* Horizontal divider */
    hr {
        border: none !important;
        border-top: 1px solid #f7d77e !important;
        opacity: 0.6;
        margin: 1.5rem 0 !important;
        box-shadow: 0 0 5px rgba(247, 215, 126, 0.3);
    }

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-top: 1px solid #f7d77e !important;
        opacity: 0.5;
        margin: 1rem 0 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Remove Streamlit branding menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header toolbar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* FORCE transparent dataframe - add at the very end */
    iframe[title="streamlit_app.components.v1.dataframe"] {
        background: transparent !important;
    }

    div[data-testid="stDataFrame"] * {
        background-color: transparent !important;
    }

    .glideDataEditor {
        background-color: transparent !important;
    }

    .dvn-scroller {
        background-color: transparent !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def render_sidebar():
    """Render the sidebar with navigation and information."""

    st.sidebar.markdown(
        '<h2 style="color: #f7d77e !important; text-shadow: 0 0 10px rgba(247, 215, 126, 0.5);">Navigation</h2>',
        unsafe_allow_html=True
    )

    # Navigation
    page = st.sidebar.radio(
        "Choose a page", ["Price Prediction", "Data Analytics"], index=0
    )

    st.sidebar.markdown("---")

    # Information box
    st.sidebar.markdown(
        """
    <div class="sidebar-info">
        <h3>About This App</h3>
        <p>
        "This application uses a trained LR model to predict used car prices "
        "based on various features."
        </p>
        <h4>How it works:</h4>
        <ul>
            <li>Enter your car's specifications</li>
            <li>Our ML model processes the data</li>
            <li>Get an instant price prediction</li>
            <li>View confidence score and insights</li>
        </ul>
        <h4>Model Features:</h4>
        <ul>
            <li>Trained on thousands of car listings</li>
            <li>Considers 11+ key factors</li>
            <li>Provides confidence scores</li>
            <li>Real-time predictions</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    return page


def render_header():
    """Render the main header."""
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="
                font-size: 3.5rem !important;
                font-weight: 900 !important;
                color: #f7d77e !important;
                letter-spacing: 2px !important;
                margin-bottom: 0.5rem !important;
            ">
                Used Car Price Predictor
            </h1>
            <p style="
                font-size: 1.3rem !important;
                color: #ffffff !important;
                text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.9) !important;
                font-weight: 300 !important;
            ">
                Get instant price predictions for used cars using advanced ML algorithms
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_prediction_page():
    """Render the prediction page."""
    render_header()

    # Create layout for prediction page
    col1, col2 = st.columns([3, 1])

    with col1:
        # Initialize and render prediction interface
        prediction_interface = PredictionInterface()
        prediction_interface.render()

    with col2:
        # Additional information or help
        with st.expander("💡 Tips & Insights", expanded=False):
            # Tips for Accurate Predictions
            st.markdown(
                '<h4 style="color: #f7d77e !important;">Tips for Accurate Predictions</h4>', unsafe_allow_html=True)
            st.markdown("""
                - Provide accurate manufacturing year
                - Be honest about accident history
                - Include complete service records
                - Consider current market conditions
            """)

            # Factors Affecting Price
            st.markdown(
                '<h4 style="color: #f7d77e !important; margin-top: 1.5rem;">Factors Affecting Price</h4>', unsafe_allow_html=True)
            st.markdown(
                """
                <ul style="color: #e0e0e0; line-height: 1.8; list-style-type: disc; margin-left: 0rem;">
                    <li><strong style="color: #f7d77e;">Age:</strong> Newer cars typically cost more
                    <li><strong style="color: #f7d77e;">Brand:</strong> Luxury brands hold value better
                    <li><strong style="color: #f7d77e;">Mileage:</strong> Better fuel efficiency increases value
                    <li><strong style="color: #f7d77e;">Accidents:</strong> History affects resale value
                    <li><strong style="color: #f7d77e;">Maintenance:</strong> Good service history helps
                </ul>
                """,
                unsafe_allow_html=True
            )


def render_analytics_page():
    """Render the data analytics page."""
    # Initialize and render data viewer
    data_viewer = DataViewer()
    data_viewer.render()


def main():
    """Main application function."""
    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Set background
    # if selected_page == "Price Prediction":
    set_background_image(use_image=True)
    # else:
    # set_background_image(use_image=False)

    # Render the selected page
    if selected_page == "Price Prediction":
        render_prediction_page()
    elif selected_page == "Data Analytics":
        render_analytics_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #00d9ff; padding: 1rem;">
            <p style="text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);">
            Built with Streamlit and FastAPI | Machine Learning in Production
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
