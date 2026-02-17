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
import smtplib
import re
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import gspread
from google.oauth2.service_account import Credentials

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
    except Exception:
        return None


def validate_api_key(entered_key: str) -> bool:
    """Validate the entered API key."""
    valid_key = get_valid_api_key()
    if valid_key is None:
        return False
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


def sanitize_sheet_tab_name(name: str) -> str:
    """
    Sanitize a string to be a valid Google Sheets tab name.

    Google Sheets tab names cannot contain: / \\ * ? : [ ]
    Max length is 100 characters.

    Args:
        name: The raw class name string

    Returns:
        A sanitized string safe for use as a Google Sheets tab name.
    """
    if not name or not name.strip():
        return "Evaluation"

    sanitized = name.strip()
    # Replace forbidden characters with underscore
    for char in ['/', '\\', '*', '?', ':', '[', ']']:
        sanitized = sanitized.replace(char, '_')

    # Truncate to 100 characters
    sanitized = sanitized[:100]

    # Final fallback if everything was stripped
    if not sanitized.strip():
        return "Evaluation"

    return sanitized


def upload_to_google_sheets(df: pd.DataFrame, class_name: str, spreadsheet_id: str = None) -> tuple:
    """
    Upload a DataFrame to a new tab in a Google Spreadsheet.

    Creates a new worksheet tab with the class name and writes the
    evaluation data. Handles duplicate tab names by appending a timestamp.

    Args:
        df: The evaluation DataFrame (potentially edited by user).
        class_name: The class name to use as the tab name.
        spreadsheet_id: The Google Spreadsheet ID to upload to.

    Returns:
        Tuple of (success: bool, message: str).
    """
    if not spreadsheet_id:
        return (False, "No center selected. Please select a center in the sidebar.")

    try:
        gc = _get_gspread_client()

        # Sanitize tab name
        tab_name = sanitize_sheet_tab_name(class_name)

        # Open spreadsheet
        spreadsheet = gc.open_by_key(spreadsheet_id)

        # Handle duplicate tab names by appending timestamp
        existing_titles = [ws.title for ws in spreadsheet.worksheets()]
        if tab_name in existing_titles:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tab_name = f"{tab_name}_{timestamp}"[:100]

        # Clean DataFrame: replace NaN with empty string, convert all to str
        df_clean = df.fillna("").astype(str)

        # Create new worksheet
        rows = len(df_clean) + 1  # +1 for header
        cols = len(df_clean.columns)
        worksheet = spreadsheet.add_worksheet(title=tab_name, rows=rows, cols=cols)

        # Prepare data as list of lists (header + data rows)
        header = df_clean.columns.tolist()
        data = df_clean.values.tolist()
        all_data = [header] + data

        # Write data to sheet
        worksheet.update(all_data, value_input_option='USER_ENTERED')

        return (True, f"✅ Successfully uploaded to sheet tab '{tab_name}'")

    except gspread.exceptions.SpreadsheetNotFound:
        return (False, "Spreadsheet not found. The sheet ID may be invalid or the service account lacks access.")
    except gspread.exceptions.APIError as e:
        return (False, f"Google Sheets API error: {str(e)}")
    except Exception as e:
        return (False, f"Failed to upload to Google Sheets: {str(e)}")


def _get_gspread_client():
    """Create and return an authenticated gspread client using service account credentials."""
    service_account_info = dict(st.secrets["gcp_service_account"])
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    return gspread.authorize(creds)


def _get_centers_file_path():
    """Return the absolute path to centers.json, located alongside this script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "centers.json")


def load_centers():
    """
    Load center configurations from centers.json.

    Returns:
        list[dict]: List of center dicts with 'sheet_id' key. Empty list if file missing/malformed.
    """
    filepath = _get_centers_file_path()
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data.get("centers", [])
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return []


def save_centers(centers):
    """
    Save center configurations to centers.json.

    Args:
        centers: List of center dicts, each with a 'sheet_id' key.
    """
    filepath = _get_centers_file_path()
    with open(filepath, "w") as f:
        json.dump({"centers": centers}, f, indent=2)


@st.cache_data(ttl=300)
def fetch_center_names(sheet_ids):
    """
    Fetch spreadsheet titles from Google for a tuple of sheet IDs.

    Cached for 5 minutes to avoid hitting the Google API on every Streamlit rerun.

    Args:
        sheet_ids: Tuple of sheet ID strings (tuple for hashability with cache).

    Returns:
        dict: Mapping of sheet_id -> spreadsheet title string.
    """
    gc = _get_gspread_client()
    names = {}
    for sid in sheet_ids:
        try:
            spreadsheet = gc.open_by_key(sid)
            names[sid] = spreadsheet.title
        except gspread.exceptions.APIError:
            names[sid] = f"Inaccessible ({sid[:8]}...)"
        except Exception:
            names[sid] = f"Unknown ({sid[:8]}...)"
    return names


def create_new_center(name):
    """
    Create a new Google Spreadsheet to serve as a center's evaluation store.

    The spreadsheet is owned by the service account and shared with the admin
    email (GMAIL_SENDER) as an editor.

    Args:
        name: Display name for the spreadsheet (becomes the title).

    Returns:
        Tuple of (success: bool, message: str, sheet_id: str).
    """
    if not name or not name.strip():
        return (False, "Center name cannot be empty.", "")

    try:
        gc = _get_gspread_client()
        spreadsheet = gc.create(name.strip())
        new_sheet_id = spreadsheet.id

        # Share with admin email so a human can access it
        try:
            admin_email = st.secrets["GMAIL_SENDER"]
            spreadsheet.share(admin_email, perm_type='user', role='writer')
        except Exception:
            pass  # Non-fatal: the service account still owns it

        # Persist to centers.json
        centers = load_centers()
        centers.append({"sheet_id": new_sheet_id})
        save_centers(centers)

        # Clear cached center names so the new one shows up
        fetch_center_names.clear()

        return (True, f"Created center '{name.strip()}' successfully.", new_sheet_id)

    except gspread.exceptions.APIError as e:
        return (False, f"Google API error: {str(e)}", "")
    except Exception as e:
        return (False, f"Failed to create center: {str(e)}", "")


def validate_email_addresses(email_string: str) -> tuple:
    """
    Validate one or more comma-separated email addresses.

    Args:
        email_string: Raw string from user input, may contain
                      comma-separated emails.

    Returns:
        Tuple of (is_valid: bool, parsed_emails: list[str], error_message: str).
    """
    if not email_string or not email_string.strip():
        return (False, [], "No email address provided.")

    # Split by comma and strip whitespace
    parts = [e.strip() for e in email_string.split(",") if e.strip()]

    if not parts:
        return (False, [], "No valid email addresses found.")

    # Simple email regex validation
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    invalid = [e for e in parts if not email_pattern.match(e)]

    if invalid:
        return (False, [], f"Invalid email address(es): {', '.join(invalid)}")

    return (True, parts, "")


def send_evaluation_email(
    to_emails: list,
    class_name: str,
    df: pd.DataFrame,
    csv_data: str,
    pdf_data: bytes
) -> tuple:
    """
    Send evaluation results via email with CSV and PDF attachments.

    Args:
        to_emails: List of validated recipient email addresses.
        class_name: The class name for subject and body.
        df: The DataFrame used to generate summary statistics.
        csv_data: CSV string content for attachment.
        pdf_data: PDF bytes content for attachment.

    Returns:
        Tuple of (success: bool, message: str).
    """
    FROM_EMAIL = st.secrets["GMAIL_SENDER"]

    try:
        APP_PASSWORD = st.secrets["GMAIL_SMTP"]
    except Exception:
        return (False, "GMAIL_SMTP password not found in secrets. Please configure it.")

    try:
        # Build subject
        subject = f"Teacher Evaluation Report - {class_name}" if class_name else "Teacher Evaluation Report"

        # Build summary statistics for email body
        total = len(df)
        completed = len(df[df.get('Status', pd.Series()).str.upper() == 'COMPLETED']) if 'Status' in df.columns else 0
        partial = len(df[df.get('Status', pd.Series()).str.upper() == 'PARTIAL']) if 'Status' in df.columns else 0
        missing = len(df[df.get('Status', pd.Series()).str.upper() == 'MISSING']) if 'Status' in df.columns else 0

        # Calculate average rating excluding zeros
        if 'Overall Rating' in df.columns:
            ratings = pd.to_numeric(df['Overall Rating'], errors='coerce')
            valid_ratings = ratings[ratings > 0]
            avg_rating = valid_ratings.mean() if len(valid_ratings) > 0 else 0
        else:
            avg_rating = 0

        body = f"""Teacher Evaluation Report
========================
Class: {class_name or "N/A"}

Summary:
- Total Activities: {total}
- Completed: {completed}
- Partial: {partial}
- Missing: {missing}
- Average Rating: {avg_rating:.1f}/10

Please find the detailed evaluation results attached as CSV and PDF files.

---
This email was sent automatically by the Teacher Evaluation Dashboard.
"""

        # Create email message
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach CSV
        csv_attachment = MIMEBase('text', 'csv')
        csv_attachment.set_payload(csv_data.encode('utf-8'))
        encoders.encode_base64(csv_attachment)
        csv_filename = f"teacher_evaluation_{class_name}.csv" if class_name else "teacher_evaluation.csv"
        csv_attachment.add_header('Content-Disposition', f'attachment; filename="{csv_filename}"')
        msg.attach(csv_attachment)

        # Attach PDF
        pdf_attachment = MIMEBase('application', 'pdf')
        pdf_attachment.set_payload(pdf_data)
        encoders.encode_base64(pdf_attachment)
        pdf_filename = f"teacher_evaluation_{class_name}.pdf" if class_name else "teacher_evaluation.pdf"
        pdf_attachment.add_header('Content-Disposition', f'attachment; filename="{pdf_filename}"')
        msg.attach(pdf_attachment)

        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=30)
        server.starttls()
        server.login(FROM_EMAIL, APP_PASSWORD)
        server.sendmail(FROM_EMAIL, to_emails, msg.as_string())
        server.quit()

        return (True, f"✅ Email sent successfully to {', '.join(to_emails)}")

    except smtplib.SMTPAuthenticationError:
        return (False, "SMTP authentication failed. Check Gmail app password in secrets.")
    except smtplib.SMTPException as e:
        return (False, f"SMTP error: {str(e)}")
    except TimeoutError:
        return (False, "SMTP connection timed out. Please try again.")
    except Exception as e:
        return (False, f"Failed to send email: {str(e)}")


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
            <div class="auth-subtitle">Please enter your Password to access the dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        
        # API Key input
        api_key_input = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your Password",
            help="Enter the Password provided to you",
            key="Password_input"
        )
        
        # Verify button
        if st.button("🔓 Verify & Continue", use_container_width=True):
            if api_key_input:
                if validate_api_key(api_key_input):
                    st.session_state.authenticated = True
                    st.session_state.api_key = api_key_input
                    st.rerun()
                else:
                    st.error("❌ Invalid Password. Please check and try again.")
            else:
                st.warning("⚠️ Please enter a Password.")
        
        st.markdown("""
        <p style="text-align: center; color: #9ca3af; font-size: 0.85rem; margin-top: 1.5rem;">
            Don't have a Password? Contact your administrator.
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
    if 'sheets_upload_status' not in st.session_state:
        st.session_state.sheets_upload_status = None
    if 'sheets_upload_message' not in st.session_state:
        st.session_state.sheets_upload_message = ""
    if 'email_send_status' not in st.session_state:
        st.session_state.email_send_status = None
    if 'email_send_message' not in st.session_state:
        st.session_state.email_send_message = ""
    if 'selected_center_id' not in st.session_state:
        st.session_state.selected_center_id = None
    if 'show_add_center_form' not in st.session_state:
        st.session_state.show_add_center_form = False
    if 'add_center_status' not in st.session_state:
        st.session_state.add_center_status = None
    if 'add_center_message' not in st.session_state:
        st.session_state.add_center_message = ""

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

        # --- Center Selection ---
        st.markdown("##### 🏢 Center")
        centers = load_centers()

        if centers:
            sheet_ids = tuple(c["sheet_id"] for c in centers)
            center_names = fetch_center_names(sheet_ids)

            display_options = []
            id_by_display = {}
            for c in centers:
                sid = c["sheet_id"]
                title = center_names.get(sid, f"Unknown ({sid[:8]}...)")
                display_options.append(title)
                id_by_display[title] = sid

            selected_display = st.selectbox(
                "Center",
                options=display_options,
                help="Select the center whose spreadsheet will receive the evaluation results.",
                label_visibility="collapsed",
                key="center_selectbox"
            )

            if selected_display:
                st.session_state.selected_center_id = id_by_display[selected_display]
                # Show clickable link to the selected sheet
                sheet_url = f"https://docs.google.com/spreadsheets/d/{id_by_display[selected_display]}"
                st.markdown(f'<a href="{sheet_url}" target="_blank" style="font-size: 0.85rem; color: #3d5a80;">📎 Open Sheet</a>', unsafe_allow_html=True)
        else:
            st.info("No centers configured. Add one below.")
            st.session_state.selected_center_id = None

        if st.button("➕ Add New Center", use_container_width=True, key="toggle_add_center"):
            st.session_state.show_add_center_form = not st.session_state.show_add_center_form
            st.session_state.add_center_status = None
            st.session_state.add_center_message = ""

        if st.session_state.show_add_center_form:
            new_center_name = st.text_input(
                "New Center Name",
                placeholder="e.g., Downtown Learning Center",
                key="new_center_name_input",
                label_visibility="collapsed"
            )
            if st.button("Create Center", use_container_width=True, key="create_center_btn"):
                if new_center_name and new_center_name.strip():
                    with st.spinner("Creating spreadsheet..."):
                        success, message, new_id = create_new_center(new_center_name)
                        st.session_state.add_center_status = "success" if success else "error"
                        st.session_state.add_center_message = message
                        if success:
                            st.session_state.selected_center_id = new_id
                            st.session_state.show_add_center_form = False
                            st.rerun()
                else:
                    st.session_state.add_center_status = "error"
                    st.session_state.add_center_message = "Please enter a name for the center."

            if st.session_state.add_center_status == "success":
                st.success(st.session_state.add_center_message)
            elif st.session_state.add_center_status == "error":
                st.error(st.session_state.add_center_message)

        st.markdown("")

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

        st.markdown("##### ✉️ Email Recipient (Optional)")
        email_recipient = st.text_input(
            "Email Address",
            placeholder="email@example.com or comma-separated",
            help="Enter email address(es) to send results. Separate multiple with commas.",
            label_visibility="collapsed",
            key="email_recipient_input"
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
            st.session_state.sheets_upload_status = None
            st.session_state.sheets_upload_message = ""
            st.session_state.email_send_status = None
            st.session_state.email_send_message = ""

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

                        st.toast("Evaluation complete! Your results are ready.", icon="✅")

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
        st.markdown("### 💾 Export & Share")

        # Row 1: File downloads
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
                file_name_pdf = "teacher_evaluation_report" + st.session_state.class_name_saved + ".pdf"
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=file_name_pdf,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

        # Row 2: Share options
        col3, col4 = st.columns(2)

        with col3:
            # Google Sheets upload
            if st.button("📊 Upload to Google Sheets", use_container_width=True, key="sheets_upload_btn"):
                selected_id = st.session_state.get("selected_center_id")
                if not selected_id:
                    st.session_state.sheets_upload_status = "error"
                    st.session_state.sheets_upload_message = "No center selected. Please select a center in the sidebar."
                else:
                    with st.spinner("Uploading to Google Sheets..."):
                        success, message = upload_to_google_sheets(
                            edited_df,
                            st.session_state.class_name_saved,
                            spreadsheet_id=selected_id
                        )
                        st.session_state.sheets_upload_status = "success" if success else "error"
                        st.session_state.sheets_upload_message = message

            # Display upload status
            if st.session_state.sheets_upload_status == "success":
                st.success(st.session_state.sheets_upload_message)
            elif st.session_state.sheets_upload_status == "error":
                st.error(st.session_state.sheets_upload_message)

        with col4:
            # Send Email
            email_input_value = st.session_state.get("email_recipient_input", "")
            email_button_disabled = not email_input_value.strip()

            if st.button(
                "✉️ Send Email",
                use_container_width=True,
                key="send_email_btn",
                disabled=email_button_disabled
            ):
                # Validate email addresses
                is_valid, parsed_emails, error_msg = validate_email_addresses(email_input_value)

                if not is_valid:
                    st.session_state.email_send_status = "error"
                    st.session_state.email_send_message = error_msg
                else:
                    with st.spinner("Sending email..."):
                        try:
                            # Generate CSV and PDF for attachment
                            csv_for_email = edited_df.to_csv(index=False)
                            pdf_for_email = generate_pdf_report(
                                edited_df,
                                class_name=st.session_state.class_name_saved
                            )

                            success, message = send_evaluation_email(
                                to_emails=parsed_emails,
                                class_name=st.session_state.class_name_saved,
                                df=edited_df,
                                csv_data=csv_for_email,
                                pdf_data=pdf_for_email
                            )
                            st.session_state.email_send_status = "success" if success else "error"
                            st.session_state.email_send_message = message
                        except Exception as e:
                            st.session_state.email_send_status = "error"
                            st.session_state.email_send_message = f"Error: {str(e)}"

            if not email_input_value.strip():
                st.caption("💡 Enter an email in the sidebar to enable sending.")

            # Display email status
            if st.session_state.email_send_status == "success":
                st.success(st.session_state.email_send_message)
            elif st.session_state.email_send_status == "error":
                st.error(st.session_state.email_send_message)

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