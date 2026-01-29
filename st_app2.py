"""
Teacher Evaluation Dashboard
A Streamlit application for evaluating teacher performance based on audio and curriculum files.
"""

import streamlit as st
import pandas as pd
import json
import time
import io
from fpdf import FPDF
from thapp import run_teacher_evaluation

# Page configuration
st.set_page_config(
    page_title="Teacher Evaluation Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Serif+Display&display=swap');
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    }
    
    /* Header styling */
    .main-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3a5f 0%, #3d5a80 50%, #5c7a99 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-family: 'DM Sans', sans-serif;
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .input-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        color: #1e3a5f;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status badges */
    .status-completed {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-partial {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-missing {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'DM Sans', sans-serif;
        background: linear-gradient(135deg, #1e3a5f 0%, #3d5a80 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 58, 95, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 58, 95, 0.4);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        font-family: 'DM Sans', sans-serif;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        padding: 0.7rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #3d5a80;
        box-shadow: 0 0 0 3px rgba(61, 90, 128, 0.1);
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid #e5e7eb;
        border-top: 4px solid #3d5a80;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-family: 'DM Sans', sans-serif;
        color: #3d5a80;
        font-size: 1.1rem;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    
    .metric-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
    
    /* Table styling */
    .dataframe {
        font-family: 'DM Sans', sans-serif !important;
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 2rem 0;
    }
    
    /* Success message */
    .success-banner {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Auth card styling */
    .auth-card {
        background: white;
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
        max-width: 450px;
        margin: 2rem auto;
        text-align: center;
    }
    
    .auth-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    
    .auth-subtitle {
        font-family: 'DM Sans', sans-serif;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    
    /* Lock icon */
    .lock-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Data editor styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# Get API key from secrets
def get_valid_api_key():
    """Get the valid API key from secrets."""
    try:
        return st.secrets["api"]["api_key"]
    except:
        return "a7k9m3x5q1"  # Fallback default


def validate_api_key(entered_key: str) -> bool:
    """Validate the entered API key."""
    valid_key = get_valid_api_key()
    return entered_key == valid_key


def convert_results_to_dataframe(results: list) -> pd.DataFrame:
    """
    Convert the evaluation results to a pandas DataFrame.
    Excludes activity_context column.
    
    Args:
        results: List of dictionaries containing evaluation results
        
    Returns:
        pandas DataFrame with the evaluation data
    """
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Remove activity_context if it exists
    if 'activity_context' in df.columns:
        df = df.drop(columns=['activity_context'])
    
    # Reorder columns for better display (excluding activity_context)
    column_order = [
        'activity_name',
        'status',
        'overall_rating',
        'clarity_rating',
        'interaction_rating',
        'pacing_rating',
        'clarity_assessment',
        'interaction_assessment',
        'pacing_assessment',
        'comments'
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Rename columns for better display
    column_rename = {
        'activity_name': 'Activity Name',
        'status': 'Status',
        'overall_rating': 'Overall Rating',
        'clarity_rating': 'Clarity Rating',
        'interaction_rating': 'Interaction Rating',
        'pacing_rating': 'Pacing Rating',
        'clarity_assessment': 'Clarity Assessment',
        'interaction_assessment': 'Interaction Assessment',
        'pacing_assessment': 'Pacing Assessment',
        'comments': 'Comments'
    }
    
    df = df.rename(columns=column_rename)
    
    return df

def generate_pdf_report(df: pd.DataFrame, class_name: str = "") -> bytes:
    """Generate a PDF report from the evaluation dataframe."""
    
    def sanitize_text_for_pdf(text):
        """Convert Unicode characters to ASCII-safe equivalents."""
        if not text or text != text:
            return ""
        
        text = str(text)
        replacements = {
            '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
            '\u201c': '"', '\u201d': '"', '\u2022': '*', '\u2026': '...',
            '\u00e9': 'e', '\u00e8': 'e', '\u00ea': 'e', '\u00fc': 'u',
            '\u00e0': 'a', '\u00e7': 'c',
        }
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        
        return text.encode('latin-1', errors='ignore').decode('latin-1')
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 20)
            self.set_text_color(30, 58, 95)
            self.cell(0, 15, 'Teacher Evaluation Report', align='C', ln=True)
            if class_name:
                self.set_font('Helvetica', '', 12)
                self.set_text_color(107, 114, 128)
                # Sanitize class name
                safe_class = sanitize_text_for_pdf(class_name)
                self.cell(0, 8, f'Class: {safe_class}', align='C', ln=True)
            self.ln(5)
            self.set_draw_color(229, 231, 235)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
    
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Summary section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 58, 95)
    pdf.cell(0, 10, 'Summary Overview', ln=True)
    pdf.ln(2)
    
    total_activities = len(df)
    completed = len(df[df['Status'] == 'COMPLETED']) if 'Status' in df.columns else 0
    partial = len(df[df['Status'] == 'PARTIAL']) if 'Status' in df.columns else 0
    missing = len(df[df['Status'] == 'MISSING']) if 'Status' in df.columns else 0
    
    if 'Overall Rating' in df.columns:
        ratings = df[df['Overall Rating'] > 0]['Overall Rating'].tolist()
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
    else:
        avg_rating = 0
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(55, 65, 81)
    
    summary_data = [
        ('Total Activities', str(total_activities)),
        ('Completed', str(completed)),
        ('Partial', str(partial)),
        ('Missing', str(missing)),
        ('Average Rating', f'{avg_rating:.1f}/10')
    ]
    
    pdf.set_fill_color(249, 250, 251)
    for i, (label, value) in enumerate(summary_data):
        fill = i % 2 == 0
        # Sanitize summary values
        safe_label = sanitize_text_for_pdf(label)
        safe_value = sanitize_text_for_pdf(value)
        pdf.cell(60, 8, safe_label, border=1, fill=fill)
        pdf.cell(30, 8, safe_value, border=1, fill=fill, ln=True)
    
    pdf.ln(10)
    
    # Activity details
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 58, 95)
    pdf.cell(0, 10, 'Activity Details', ln=True)
    pdf.ln(5)
    
    for idx, row in df.iterrows():
        if pdf.get_y() > 230:
            pdf.add_page()
        
        # Activity name header - SANITIZED
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(30, 58, 95)
        activity_name = sanitize_text_for_pdf(row.get('Activity Name', 'Unknown Activity'))
        pdf.cell(0, 8, activity_name, ln=True)
        
        # Status - SANITIZED
        status = sanitize_text_for_pdf(row.get('Status', 'N/A'))
        pdf.set_font('Helvetica', 'B', 9)
        if status == 'COMPLETED':
            pdf.set_text_color(16, 185, 129)
        elif status == 'PARTIAL':
            pdf.set_text_color(245, 158, 11)
        else:
            pdf.set_text_color(239, 68, 68)
        pdf.cell(0, 6, f'Status: {status}', ln=True)
        
        # Ratings - SANITIZED
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(55, 65, 81)
        
        ratings_text = []
        if 'Overall Rating' in row:
            ratings_text.append(f"Overall: {row['Overall Rating']}/10")
        if 'Clarity Rating' in row:
            ratings_text.append(f"Clarity: {row['Clarity Rating']}/10")
        if 'Interaction Rating' in row:
            ratings_text.append(f"Interaction: {row['Interaction Rating']}/10")
        if 'Pacing Rating' in row:
            ratings_text.append(f"Pacing: {row['Pacing Rating']}/10")
        
        if ratings_text:
            safe_ratings = sanitize_text_for_pdf('  |  '.join(ratings_text))
            pdf.cell(0, 6, safe_ratings, ln=True)
        
        pdf.ln(3)
        
        # Assessments - SANITIZED
        assessments = [
            ('Clarity Assessment', row.get('Clarity Assessment', '')),
            ('Interaction Assessment', row.get('Interaction Assessment', '')),
            ('Pacing Assessment', row.get('Pacing Assessment', '')),
            ('Comments', row.get('Comments', ''))
        ]
        
        for label, content in assessments:
            if content and str(content) != 'nan' and str(content).strip():
                pdf.set_font('Helvetica', 'B', 9)
                pdf.set_text_color(75, 85, 99)
                safe_label = sanitize_text_for_pdf(label)
                pdf.cell(0, 5, safe_label + ':', ln=True)
                
                pdf.set_font('Helvetica', '', 9)
                pdf.set_text_color(107, 114, 128)
                
                # Sanitize content before adding to PDF
                content_str = sanitize_text_for_pdf(str(content))
                pdf.multi_cell(0, 5, content_str)
                pdf.ln(2)
        
        # Separator line
        pdf.set_draw_color(229, 231, 235)
        pdf.set_line_width(0.3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)
    
    return pdf.output(dest='S').encode('latin-1')


def get_status_color(status: str) -> str:
    """Return color based on status."""
    status_colors = {
        'COMPLETED': '#10b981',
        'PARTIAL': '#f59e0b',
        'MISSING': '#ef4444'
    }
    return status_colors.get(status.upper(), '#6b7280')


def show_auth_page():
    """Display the authentication page."""
    st.markdown('<h1 class="main-header">📚 Teacher Evaluation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive analysis of teaching performance across activities</p>', unsafe_allow_html=True)
    
    # Center the auth card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="auth-card">
            <div class="lock-icon">🔐</div>
            <div class="auth-title">Authentication Required</div>
            <div class="auth-subtitle">Please enter your API key to access the dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        
        # API Key input
        api_key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your API key",
            help="Enter the API key provided to you",
            key="api_key_input"
        )
        
        # Verify button
        if st.button("🔓 Verify & Continue", use_container_width=True):
            if api_key_input:
                if validate_api_key(api_key_input):
                    st.session_state.authenticated = True
                    st.session_state.api_key = api_key_input
                    st.rerun()
                else:
                    st.error("❌ Invalid API key. Please check and try again.")
            else:
                st.warning("⚠️ Please enter an API key.")
        
        st.markdown("""
        <p style="text-align: center; color: #9ca3af; font-size: 0.85rem; margin-top: 1.5rem;">
            Don't have an API key? Contact your administrator.
        </p>
        """, unsafe_allow_html=True)


def show_main_app():
    """Display the main application after authentication."""
    # Header
    st.markdown('<h1 class="main-header">📚 Teacher Evaluation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive analysis of teaching performance across activities</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'class_name_saved' not in st.session_state:
        st.session_state.class_name_saved = ""
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### ⚙️ Evaluation Settings")
        
        
        # Input fields
        st.markdown("##### 🎵 Audio File URL")
        audio_url = st.text_input(
            "Audio File URL",
            placeholder="https://drive.google.com/file/d/.../view",
            help="Enter the Google Drive URL for the audio recording",
            label_visibility="collapsed"
        )
        
        st.markdown("##### 📄 Curriculum URL")
        curriculum_url = st.text_input(
            "Curriculum URL",
            placeholder="https://drive.google.com/file/d/.../view",
            help="Enter the Google Drive URL for the curriculum document",
            label_visibility="collapsed"
        )
        
        st.markdown("##### 👥 Number of Students")
        num_students = st.number_input(
            "Number of Students",
            min_value=1,
            max_value=100,
            value=1,
            help="Enter the number of students in the class",
            label_visibility="collapsed"
        )
        
        st.markdown("##### 🏫 Class Name")
        class_name = st.text_input(
            "Class Name",
            placeholder="e.g., Class 1, Toddler Group A",
            help="Enter the name of the class",
            label_visibility="collapsed"
        )
        
        st.markdown("##### 📝 Custom Instructions")
        custom_instruction = st.text_area(
            "Custom Instructions",
            placeholder="e.g., more lenient evaluation, focus on engagement...",
            help="Enter any custom instructions for the evaluation",
            height=100,
            label_visibility="collapsed"
        )
        

        
        # Run evaluation button
        run_evaluation = st.button("🚀 Run Evaluation", use_container_width=True)
        st.markdown("---")

        st.success("✅ Authenticated")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.api_key = None
            st.rerun()
        
        st.markdown("---")
    
    # Main content area
    if run_evaluation:
        # Validate inputs
        if not audio_url or not curriculum_url:
            st.error("⚠️ Please provide both Audio File URL and Curriculum URL to proceed.")
        else:
            # Save class name for PDF
            st.session_state.class_name_saved = class_name
            
            # Show loading animation
            with st.container():
                loading_placeholder = st.empty()
                
                with loading_placeholder.container():
                    st.markdown("""
                    <div class="loading-container">
                        <div class="loading-spinner"></div>
                        <div class="loading-text">Analyzing teaching performance...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress messages
                    progress_messages = [
                        "🎧 Processing audio file...",
                        "📖 Analyzing curriculum alignment...",
                        "📊 Evaluating activity performance...",
                        "✨ Generating insights..."
                    ]
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progress while processing
                    for i, msg in enumerate(progress_messages):
                        status_text.text(msg)
                        progress_bar.progress((i + 1) * 25)
                        time.sleep(0.5)
                    
                    # Run the actual evaluation
                    try:
                        results = run_teacher_evaluation(
                            audio_file_drive_url=audio_url,
                            curriculum_drive_url=curriculum_url,
                            num_students=num_students,
                            custom_instruction=custom_instruction,
                            class_name=class_name
                        )
                        
                        # Store results in session state
                        st.session_state.results = results
                        st.session_state.df = convert_results_to_dataframe(results)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Evaluation complete!")
                        time.sleep(0.5)
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred during evaluation: {str(e)}")
                        st.session_state.results = None
                        st.session_state.df = None
                
                # Clear loading animation
                loading_placeholder.empty()
    
    # Display results if available
    if st.session_state.df is not None:
        df = st.session_state.df
        results = st.session_state.results
        
        # Success message
        st.markdown("""
        <div class="success-banner">
            ✅ Evaluation completed successfully! Review and edit the results below.
        </div>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        st.markdown("### 📊 Summary Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_activities = len(df)
            st.metric("Total Activities", total_activities)
        
        with col2:
            completed = len([r for r in results if r.get('status', '').upper() == 'COMPLETED'])
            st.metric("Completed", completed)
        
        with col3:
            partial = len([r for r in results if r.get('status', '').upper() == 'PARTIAL'])
            st.metric("Partial", partial)
        
        with col4:
            missing = len([r for r in results if r.get('status', '').upper() == 'MISSING'])
            st.metric("Missing", missing)
        
        with col5:
            # Calculate average rating excluding zeros
            ratings = [r.get('overall_rating', 0) for r in results if r.get('overall_rating', 0) > 0]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            st.metric("Avg Rating", f"{avg_rating:.1f}/10")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Editable data table
        st.markdown("### ✏️ Evaluation Results (Editable)")
        st.markdown("*Click on any cell to edit. Changes are automatically saved.*")
        
        # Column configuration for the data editor
        column_config = {
            "Activity Name": st.column_config.TextColumn(
                "Activity Name",
                width="medium",
                help="Name of the evaluated activity"
            ),
            "Status": st.column_config.SelectboxColumn(
                "Status",
                width="small",
                options=["COMPLETED", "PARTIAL", "MISSING"],
                help="Completion status of the activity"
            ),
            "Overall Rating": st.column_config.NumberColumn(
                "Overall Rating",
                width="small",
                min_value=0,
                max_value=10,
                step=0.1,
                format="%.1f",
                help="Overall rating (0-10)"
            ),
            "Clarity Rating": st.column_config.NumberColumn(
                "Clarity Rating",
                width="small",
                min_value=0,
                max_value=10,
                step=1,
                help="Clarity of instruction rating (0-10)"
            ),
            "Interaction Rating": st.column_config.NumberColumn(
                "Interaction Rating",
                width="small",
                min_value=0,
                max_value=10,
                step=1,
                help="Student interaction rating (0-10)"
            ),
            "Pacing Rating": st.column_config.NumberColumn(
                "Pacing Rating",
                width="small",
                min_value=0,
                max_value=10,
                step=1,
                help="Activity pacing rating (0-10)"
            ),
            "Clarity Assessment": st.column_config.TextColumn(
                "Clarity Assessment",
                width="large",
                help="Detailed assessment of clarity"
            ),
            "Interaction Assessment": st.column_config.TextColumn(
                "Interaction Assessment",
                width="large",
                help="Detailed assessment of interactions"
            ),
            "Pacing Assessment": st.column_config.TextColumn(
                "Pacing Assessment",
                width="large",
                help="Detailed assessment of pacing"
            ),
            "Comments": st.column_config.TextColumn(
                "Comments",
                width="large",
                help="Additional comments and recommendations"
            ),
        }
        
        # Display editable dataframe
        edited_df = st.data_editor(
            df,
            column_config=column_config,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            key="evaluation_editor"
        )
        
        # Update session state with edited data
        st.session_state.df = edited_df
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Export options
        st.markdown("### 💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as CSV
            csv = edited_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="teacher_evaluation_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export as PDF
            try:
                pdf_bytes = generate_pdf_report(
                    edited_df, 
                    class_name=st.session_state.class_name_saved
                )
                file_name_pdf="teacher_evaluation_report"+st.session_state.class_name_saved+".pdf"
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=file_name_pdf,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Detailed view section
        st.markdown("### 🔍 Detailed Activity View")
        
        # Activity selector
        activity_names = edited_df['Activity Name'].tolist()
        selected_activity = st.selectbox(
            "Select an activity to view details:",
            activity_names,
            key="activity_selector"
        )
        
        if selected_activity:
            # Get the selected activity data
            activity_data = edited_df[edited_df['Activity Name'] == selected_activity].iloc[0]
            
            # Display detailed view
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Ratings")
                
                # Status badge
                status = activity_data.get('Status', 'N/A')
                status_class = f"status-{status.lower()}" if status else "status-missing"
                st.markdown(f'<span class="{status_class}">{status}</span>', unsafe_allow_html=True)
                
                st.markdown("")
                
                # Ratings display
                metrics = [
                    ("Overall", activity_data.get('Overall Rating', 0)),
                    ("Clarity", activity_data.get('Clarity Rating', 0)),
                    ("Interaction", activity_data.get('Interaction Rating', 0)),
                    ("Pacing", activity_data.get('Pacing Rating', 0))
                ]
                
                for metric_name, metric_value in metrics:
                    if metric_value and metric_value > 0:
                        # Create a progress bar style display
                        progress_pct = float(metric_value) / 10
                        color = '#10b981' if progress_pct >= 0.7 else '#f59e0b' if progress_pct >= 0.5 else '#ef4444'
                        st.markdown(f"""
                        <div style="margin-bottom: 0.8rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                                <span style="font-weight: 500; color: #374151;">{metric_name}</span>
                                <span style="font-weight: 700; color: {color};">{metric_value}/10</span>
                            </div>
                            <div style="background: #e5e7eb; border-radius: 10px; height: 8px; overflow: hidden;">
                                <div style="background: {color}; width: {progress_pct*100}%; height: 100%; border-radius: 10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{metric_name}:** N/A")
            
            with col2:
                st.markdown("#### Assessments")
                
                # Display assessments in expandable sections
                if activity_data.get('Clarity Assessment'):
                    with st.expander("🎯 Clarity Assessment", expanded=True):
                        st.write(activity_data['Clarity Assessment'])
                
                if activity_data.get('Interaction Assessment'):
                    with st.expander("👥 Interaction Assessment", expanded=False):
                        st.write(activity_data['Interaction Assessment'])
                
                if activity_data.get('Pacing Assessment'):
                    with st.expander("⏱️ Pacing Assessment", expanded=False):
                        st.write(activity_data['Pacing Assessment'])
                
                if activity_data.get('Comments'):
                    with st.expander("💬 Comments & Recommendations", expanded=True):
                        st.write(activity_data['Comments'])
    
    else:
        # No results yet - show instructions
        st.markdown("""
        <div class="input-card">
            <div class="card-title">📝 Getting Started</div>
            <p style="color: #6b7280; font-family: 'DM Sans', sans-serif;">
                Welcome to the Teacher Evaluation Dashboard! Follow these steps to get started:
            </p>
            <ol style="color: #374151; font-family: 'DM Sans', sans-serif; line-height: 1.8;">
                <li><strong>Enter Audio URL:</strong> Provide a Google Drive link to the class recording</li>
                <li><strong>Enter Curriculum URL:</strong> Provide a Google Drive link to the curriculum document</li>
                <li><strong>Set Parameters:</strong> Specify the number of students and class name</li>
                <li><strong>Add Instructions:</strong> (Optional) Include custom evaluation criteria</li>
                <li><strong>Run Evaluation:</strong> Click the button to start the analysis</li>
            </ol>
            <p style="color: #6b7280; font-family: 'DM Sans', sans-serif; margin-top: 1rem;">
                Once complete, you'll be able to view, edit, and export the evaluation results.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data preview
        st.markdown("### 📋 Sample Output Preview")
        st.info("This is a preview of what the evaluation results will look like. Run an evaluation to see actual data.")
        
        # Show sample data from the document
        sample_data = [
            {
                "Activity Name": "WELCOME CIRCLE & AFFIRMATIONS",
                "Status": "COMPLETED",
                "Overall Rating": 8.0,
                "Clarity Rating": 9,
                "Interaction Rating": 8,
                "Pacing Rating": 7
            },
            {
                "Activity Name": "BREATHING EXERCISES",
                "Status": "PARTIAL",
                "Overall Rating": 6.0,
                "Clarity Rating": 7,
                "Interaction Rating": 6,
                "Pacing Rating": 5
            },
            {
                "Activity Name": "CLASSICAL MUSIC LISTENING",
                "Status": "COMPLETED",
                "Overall Rating": 8.3,
                "Clarity Rating": 9,
                "Interaction Rating": 8,
                "Pacing Rating": 8
            }
        ]
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True, hide_index=True)


def main():
    """Main application entry point."""
    # Initialize authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    
    # Check authentication
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        show_main_app()


if __name__ == "__main__":
    main()