import os
import base64
import json
import time
import spacy
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from dotenv import load_dotenv
from html import escape, unescape
import google.generativeai as genai
import faiss
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import tempfile
from bs4 import BeautifulSoup
import re
import pyttsx3
import speech_recognition as sr
from gtts import gTTS
import io
import tempfile
import threading

# Load environment variables
load_dotenv()

# Initialize conversation history
conversation_history = []

# Initialize text-to-speech engine
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
except Exception as e:
    print(f"Error initializing text-to-speech engine: {str(e)}")
    engine = None

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')  # Use environment variable for secret key
app.config.update({
    'SESSION_COOKIE_NAME': 'gmail-auth-session',
    'PERMANENT_SESSION_LIFETIME': 600,  # 10 minutes
    'SESSION_COOKIE_SECURE': False,      # For local development
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'UPLOAD_FOLDER': 'static'  # Add this line to configure the upload folder
})

# OAuth configuration
CLIENT_SECRETS_FILE = "credentials.json"
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 
          'https://www.googleapis.com/auth/gmail.modify',
          'https://www.googleapis.com/auth/gmail.labels']
REDIRECT_URI = 'http://localhost:5000/callback'

# Initialize OAuth flow
flow = InstalledAppFlow.from_client_secrets_file(
    CLIENT_SECRETS_FILE,
    scopes=SCOPES,
    redirect_uri=REDIRECT_URI
)

# Initialize Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
faiss_index = faiss.IndexFlatL2(dimension)

# Store email data and embeddings
email_data = []
email_embeddings = []

# Add this near the top with other global variables
emails_processed = False

# Initialize speech recognition
recognizer = sr.Recognizer()

class RateLimiter:
    def __init__(self, max_requests_per_minute):
        self.max_requests = max_requests_per_minute
        self.requests = []
        
    def wait_if_needed(self):
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests:
            # Wait until the oldest request is more than 1 minute old
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.requests = self.requests[1:]
        
        self.requests.append(now)

# Create a global rate limiter for Gemini API
gemini_limiter = RateLimiter(max_requests_per_minute=15)

def get_header(headers, name):
    """Get header value from email headers"""
    for header in headers:
        if header['name'].lower() == name.lower():
            return header['value']
    return ''

def get_email_body(payload):
    """Extract email body from payload"""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data', '')
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8')
    elif 'body' in payload and 'data' in payload['body']:
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    return ''

def get_email_summary(payload):
    """Get email summary from payload"""
    body = get_email_body(payload)
    if not body:
        return ''
    
    try:
        gemini_limiter.wait_if_needed()
        prompt = f"Summarize this email in one line: {body}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ''

def get_email_content(service, message_id):
    """Get full email content including body"""
    message = service.users().messages().get(
        userId='me',
        id=message_id,
        format='full'
    ).execute()
    
    # Extract headers
    headers = {h['name']: h['value'] for h in message['payload']['headers']}
    
    # Extract body
    body = ""
    if 'parts' in message['payload']:
        for part in message['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                break
    elif 'body' in message['payload'] and 'data' in message['payload']['body']:
        body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')
    
    return {
        'headers': headers,
        'body': body
    }

def create_nested_label(service, parent_name, child_name):
    """Create a nested label under a parent label"""
    try:
        # First, ensure parent label exists
        parent_label = get_or_create_label(service, parent_name)
        
        # Create child label with parent reference
        label = {
            'name': f'{parent_name}/{child_name}',
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show'
        }
        
        # Check if child label already exists
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        
        for existing_label in labels:
            if existing_label['name'] == label['name']:
                return existing_label['id']
        
        # Create new nested label
        created_label = service.users().labels().create(userId='me', body=label).execute()
        return created_label['id']
    
    except Exception as e:
        print(f"Error creating nested label: {str(e)}")
        return None

def ensure_priority_labels(service):
    """Ensure all priority labels exist under the Priority parent"""
    parent_label = 'Priority'
    priority_levels = ['high-priority', 'med-priority', 'low-priority']
    
    # First ensure parent label exists
    get_or_create_label(service, parent_label)
    
    # Then create nested priority labels
    for level in priority_levels:
        create_nested_label(service, parent_label, level)

def ensure_deletion_labels(service):
    """Ensure all deletion category labels exist"""
    parent_label = 'Potential Deletion'
    deletion_categories = [
        'outdated',
        'duplicates',
        'otp',
        'account-creation',
        'delivery-confirmation',
        'promotional',
        'notifications'
    ]
    
    # First ensure parent label exists
    get_or_create_label(service, parent_label)
    
    # Then create nested deletion category labels
    for category in deletion_categories:
        create_nested_label(service, parent_label, category)

def analyze_email_with_gemini(email_content, existing_labels):
    """Analyze email content and suggest labels using Gemini"""
    gemini_limiter.wait_if_needed()
    
    # Convert empty list to empty string for the prompt
    labels_str = ', '.join(existing_labels) if existing_labels else 'No existing labels'
    
    prompt = f"""
    Analyze this email and provide two pieces of information:
    1. A single-word topic label
    2. A priority level (high-priority, med-priority, or low-priority)
    
    Existing custom labels: {labels_str}
    
    Email Subject: {email_content['headers'].get('Subject', '')}
    Email Body: {email_content['body']}
    
    For the topic label:
    - If any existing labels match the topic, use that label
    - Otherwise, suggest a new single-word label
    
    For priority:
    - high-priority: Urgent matters, time-sensitive, critical updates
    - med-priority: Important but not urgent, regular business
    - low-priority: Non-urgent, informational, newsletters
    
    Return the response in this exact format:
    TOPIC: [single word]
    PRIORITY: [high-priority/med-priority/low-priority]
    """
    
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Parse the response
    topic = None
    priority = None
    for line in response_text.split('\n'):
        if line.startswith('TOPIC:'):
            topic = line.split('TOPIC:')[1].strip()
        elif line.startswith('PRIORITY:'):
            priority = line.split('PRIORITY:')[1].strip()
    
    return topic, priority

def analyze_deletion_category(email_content):
    """Analyze if email belongs to a deletion category"""
    gemini_limiter.wait_if_needed()
    
    prompt = f"""
    Analyze this email and determine if it belongs to any of these deletion categories:
    - outdated: Old information, expired content
    - duplicates: Repeated emails, multiple copies
    - otp: One-time passwords, verification codes
    - account-creation: Account setup emails
    - delivery-confirmation: Package/order delivery updates
    - promotional: Marketing emails, advertisements
    - notifications: System notifications, automated alerts
    
    Email Subject: {email_content['headers'].get('Subject', '')}
    Email Body: {email_content['body']}
    
    If the email matches any category, return only the category name.
    If it doesn't match any category, return "none".
    """
    
    response = model.generate_content(prompt)
    category = response.text.strip().lower()
    return category if category != 'none' else None

def get_or_create_label(service, label_name):
    """Get existing label or create new one"""
    try:
        # Get all labels
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        
        # Filter out system labels (only keep custom user labels)
        custom_labels = [label for label in labels if label['type'] == 'user']
        
        # Check if label exists
        for label in custom_labels:
            if label['name'].lower() == label_name.lower():
                return label['id']
        
        # Create new label if not exists
        label = {
            'name': label_name,
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show'
        }
        created_label = service.users().labels().create(userId='me', body=label).execute()
        return created_label['id']
    
    except Exception as e:
        print(f"Error managing label: {str(e)}")
        return None

def apply_label_to_email(service, message_id, label_id):
    """Apply label to email"""
    try:
        service.users().messages().modify(
            userId='me',
            id=message_id,
            body={'addLabelIds': [label_id]}
        ).execute()
        return True
    except Exception as e:
        print(f"Error applying label: {str(e)}")
        return False

def generate_email_summaries(email_content):
    """Generate both short and detailed summaries of email content"""
    gemini_limiter.wait_if_needed()
    
    prompt = f"""
    Analyze this email and provide two summaries:
    1. A single-line summary (max 15 words)
    2. A detailed 50-word summary with key points in bullets (max 3-4 bullets)

    Email Subject: {email_content['headers'].get('Subject', '')}
    Email Body: {email_content['body']}

    Return the response in this exact format:
    SHORT: [single-line summary]
    DETAILED:
    - [bullet point 1]
    - [bullet point 2]
    - [bullet point 3]
    """
    
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Parse the response
    parts = response_text.split('DETAILED:')
    short_summary = parts[0].replace('SHORT:', '').strip()
    detailed_summary = parts[1].strip()
    
    return {
        'short': short_summary,
        'detailed': detailed_summary
    }

def generate_suggested_reply(email_content):
    """Generate a suggested reply for the email"""
    gemini_limiter.wait_if_needed()
    
    prompt = f"""
    Generate a professional and contextually appropriate reply to this email:

    Email Subject: {email_content['headers'].get('Subject', '')}
    Email Body: {email_content['body']}

    The reply should be:
    - Professional and courteous
    - Relevant to the email content
    - Concise but complete
    - In a natural, conversational tone

    Return only the reply text, nothing else.
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()

def get_embedding(text):
    """Generate embedding for text using Sentence Transformer"""
    try:
        embedding = embedding_model.encode(text, convert_to_tensor=False)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def index_email(email):
    """Index an email for search"""
    # Combine subject and body for embedding
    text = f"{email['subject']} {email['body']}"
    embedding = get_embedding(text)
    if embedding is not None:
        email_data.append(email)
        email_embeddings.append(embedding)
        # Update FAISS index
        faiss_index.add(np.array([embedding], dtype=np.float32))

def search_emails(query, k=3):
    """Search emails using FAISS"""
    global email_data, email_embeddings, faiss_index
    
    if not email_data:
        # If no emails are indexed, fetch and index them first
        try:
            credentials = flow.credentials
            service = build('gmail', 'v1', credentials=credentials)
            result = service.users().messages().list(userId='me', maxResults=50).execute()
            messages = result.get('messages', [])
            
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                headers = {h['name']: h['value'] for h in msg['payload']['headers']}
                
                # Get existing labels
                existing_label_ids = msg.get('labelIds', [])
                labels_result = service.users().labels().list(userId='me').execute()
                all_labels = labels_result.get('labels', [])
                label_id_to_name = {label['id']: label['name'] for label in all_labels}
                existing_labels = [label_id_to_name.get(lid) for lid in existing_label_ids]
                
                # Extract priority and deletion categories from existing labels
                priority_label = next((label for label in existing_labels if label.startswith('Priority/')), None)
                deletion_category = next((label.split('/')[-1] for label in existing_labels if label.startswith('Potential Deletion/')), None)
                topic_label = next((label for label in existing_labels 
                                  if not label.startswith('Priority/') 
                                  and not label.startswith('Potential Deletion/')
                                  and label not in ['Priority', 'Potential Deletion', 'Subscriptions']), None)
                
                email_info = {
                    'id': message['id'],
                    'subject': get_header(msg['payload']['headers'], 'Subject'),
                    'from': get_header(msg['payload']['headers'], 'From'),
                    'date': get_header(msg['payload']['headers'], 'Date'),
                    'body': get_email_body(msg['payload']),
                    'summary': get_email_summary(msg['payload']),
                    'priority_label': priority_label,
                    'deletion_category': deletion_category,
                    'topic_label': topic_label,
                    'short_summary': get_email_summary(msg['payload'])
                }
                index_email(email_info)
        except Exception as e:
            print(f"Error indexing emails: {e}")
            return []
    
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []
    
    # Search in FAISS index
    distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k)
    
    # Get results
    results = []
    for idx in indices[0]:
        if idx < len(email_data):
            results.append(email_data[idx])
    return results

@app.route('/')
def index():
    """Home page with login button"""
    if 'credentials' in session:
        return redirect('/emails')
    return render_template('login.html')

@app.route('/authorize')
def authorize():
    """Initiate Google OAuth flow"""
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='select_account'
    )
    session.permanent = True
    session['oauth_state'] = state
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    """Handle OAuth callback"""
    if 'oauth_state' not in session:
        return redirect('/authorize')
    
    if 'state' not in request.args:
        return redirect('/authorize')
    
    if session['oauth_state'] != request.args.get('state'):
        session.pop('oauth_state', None)
        return redirect('/authorize')
    
    try:
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        session['credentials'] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'scopes': credentials.scopes
        }
        session.pop('oauth_state', None)
        
    except Exception as e:
        return f"Authentication failed: {str(e)}", 400
    
    return redirect('/emails')

@app.route('/emails')
def show_emails():
    """Display emails page with labels"""
    if 'credentials' not in session:
        return redirect('/')
    
    try:
        global emails_processed
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        
        # Ensure all label categories exist
        ensure_priority_labels(service)
        ensure_deletion_labels(service)
        
        # Get all labels and filter out system labels
        labels_result = service.users().labels().list(userId='me').execute()
        all_labels = labels_result.get('labels', [])
        user_labels = [label for label in all_labels if label.get('type') == 'user']
        
        # Create a map of label IDs to names for quick lookup
        label_id_to_name = {label['id']: label['name'] for label in all_labels}
        
        # Retrieve emails (excluding sent emails)
        result = service.users().messages().list(
            userId='me',
            maxResults=3,  # Reduced from 10 to 5 to stay within rate limits
            q='-in:sent'  # Exclude sent emails
        ).execute()
        
        messages = []
        for msg in result.get('messages', []):
            email = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = {h['name']: h['value'] for h in email['payload']['headers']}
            
            # Get existing labels for this email
            existing_label_ids = email.get('labelIds', [])
            existing_labels = [label_id_to_name.get(lid) for lid in existing_label_ids]
            
            # Only process emails if they haven't been processed before
            if not emails_processed:
                email_content = {
                    'headers': headers,
                    'body': get_email_body(email['payload'])
                }
                
                # Generate summary
                summaries = generate_email_summaries(email_content)
                
                # Check if email already has priority and topic labels
                has_priority = any(label.startswith('Priority/') for label in existing_labels)
                has_deletion = any(label.startswith('Potential Deletion/') for label in existing_labels)
                
                if not has_priority:
                    # Analyze email for topic and priority
                    topic_label, priority_level = analyze_email_with_gemini(email_content, [l['name'] for l in user_labels])
                    
                    # Apply labels
                    if topic_label:
                        topic_label_id = get_or_create_label(service, topic_label)
                        if topic_label_id:
                            apply_label_to_email(service, msg['id'], topic_label_id)
                            existing_labels.append(topic_label)
                    
                    if priority_level:
                        priority_label = f'Priority/{priority_level}'
                        priority_label_id = create_nested_label(service, 'Priority', priority_level)
                        if priority_label_id:
                            apply_label_to_email(service, msg['id'], priority_label_id)
                            existing_labels.append(priority_label)
                
                if not has_deletion:
                    # Analyze for deletion category
                    deletion_category = analyze_deletion_category(email_content)
                    if deletion_category:
                        deletion_label = f'Potential Deletion/{deletion_category}'
                        deletion_label_id = create_nested_label(service, 'Potential Deletion', deletion_category)
                        if deletion_label_id:
                            apply_label_to_email(service, msg['id'], deletion_label_id)
                            existing_labels.append(deletion_label)
            else:
                # If emails are already processed, just get the summary
                email_content = {
                    'headers': headers,
                    'body': get_email_body(email['payload'])
                }
                summaries = generate_email_summaries(email_content)
            
            # Extract priority and deletion categories from existing labels
            priority_label = next((label.split('/')[-1] for label in existing_labels if label.startswith('Priority/')), None)
            deletion_category = next((label.split('/')[-1] for label in existing_labels if label.startswith('Potential Deletion/')), None)
            topic_label = next((label for label in existing_labels 
                              if not label.startswith('Priority/') 
                              and not label.startswith('Potential Deletion/')
                              and label not in ['Priority', 'Potential Deletion', 'Subscriptions']), None)
            
            messages.append({
                'id': msg['id'],
                'from': headers.get('From', 'Unknown'),
                'subject': headers.get('Subject', '(No Subject)'),
                'date': headers.get('Date', 'Unknown Date'),
                'topic_label': topic_label,
                'priority_label': priority_label,
                'deletion_category': deletion_category,
                'short_summary': summaries['short']
            })
        
        # Set the flag to True after processing all emails
        if not emails_processed:
            emails_processed = True
        
        # Get deletion categories for the template
        deletion_categories = [
            label['name'] for label in user_labels 
            if label['name'].startswith('Potential Deletion/')
        ]
        
        return render_template(
            'emails.html',
            emails=messages,
            labels=[l['name'] for l in user_labels],
            deletion_categories=deletion_categories
        )
    
    except Exception as e:
        session.clear()
        return f"Error fetching emails: {str(e)}", 500

@app.route('/get_detailed_summary/<message_id>')
def get_detailed_summary(message_id):
    """API endpoint to get detailed summary for an email"""
    if 'credentials' not in session:
        return {'error': 'Not authenticated'}, 401
    
    try:
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        email_content = get_email_content(service, message_id)
        summaries = generate_email_summaries(email_content)
        return {'detailed_summary': summaries['detailed']}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/logout')
def logout():
    """Clear session and logout"""
    global emails_processed
    emails_processed = False  # Reset the flag when logging out
    session.clear()
    return redirect('/')

@app.route('/generate_reply/<message_id>')
def get_suggested_reply(message_id):
    """API endpoint to get suggested reply for an email"""
    if 'credentials' not in session:
        return {'error': 'Not authenticated'}, 401
    
    try:
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        email_content = get_email_content(service, message_id)
        suggested_reply = generate_suggested_reply(email_content)
        return {'reply': suggested_reply}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/send_reply/<message_id>', methods=['POST'])
def send_reply(message_id):
    """API endpoint to send a reply to an email"""
    if 'credentials' not in session:
        return {'error': 'Not authenticated'}, 401
    
    try:
        reply_text = request.json.get('reply_text')
        if not reply_text:
            return {'error': 'Reply text is required'}, 400
        
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        
        # Get original email to extract necessary headers
        email_content = get_email_content(service, message_id)
        headers = email_content['headers']
        
        # Create reply email
        message = MIMEText(reply_text)
        message['to'] = headers.get('From')
        message['subject'] = f"Re: {headers.get('Subject', '(No Subject)')}"
        
        # Encode the message
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        # Send the reply
        service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()
        
        return {'success': True}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/delete_email/<message_id>', methods=['POST'])
def delete_email(message_id):
    """API endpoint to move an email to trash"""
    if 'credentials' not in session:
        return {'error': 'Not authenticated'}, 401
    
    try:
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        
        # Move the message to trash
        service.users().messages().trash(
            userId='me',
            id=message_id
        ).execute()
        
        return {'success': True}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/get_ner_content/<message_id>')
def get_ner_content(message_id):
    """API endpoint to get email content with NER highlighting"""
    if 'credentials' not in session:
        return {'error': 'Not authenticated'}, 401
    
    try:
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        email_content = get_email_content(service, message_id)
        
        # Clean the email body text
        body_text = email_content['body']
        
        # Remove HTML tags using BeautifulSoup
        soup = BeautifulSoup(body_text, 'html.parser')
        clean_text = soup.get_text(separator=' ', strip=True)
        
        # Remove extra whitespace and normalize line breaks
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Unescape HTML entities
        clean_text = unescape(clean_text)
        
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")
        
        # Process the cleaned text
        doc = nlp(clean_text)
        
        # Create HTML with highlighted entities
        html_content = []
        current_pos = 0
        
        for ent in doc.ents:
            # Add text before the entity
            if ent.start_char > current_pos:
                html_content.append(escape(clean_text[current_pos:ent.start_char]))
            
            # Add the highlighted entity
            html_content.append(f'<span class="ner-entity" data-entity-type="{ent.label_}">{escape(ent.text)}</span>')
            current_pos = ent.end_char
        
        # Add remaining text
        if current_pos < len(clean_text):
            html_content.append(escape(clean_text[current_pos:]))
        
        return {
            'content': ''.join(html_content),
            'entities': [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        }
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query', '')
        if not query:
            return render_template('emails.html', emails=[], search_query='')
        results = search_emails(query)
        return render_template('emails.html', emails=results, search_query=query)
    return render_template('emails.html')

def get_attachment_type(filename):
    """Determine attachment type based on filename"""
    ext = filename.lower().split('.')[-1]
    if ext in ['pdf']:
        return 'pdf'
    elif ext in ['doc', 'docx']:
        return 'word'
    elif ext in ['jpg', 'jpeg', 'png', 'gif']:
        return 'image'
    return 'other'

def get_attachments(service):
    """Get all attachments from emails with size information"""
    try:
        result = service.users().messages().list(userId='me', maxResults=50).execute()
        messages = result.get('messages', [])
        
        pdf_attachments = []
        word_attachments = []
        image_attachments = []
        sender_sizes = defaultdict(int)  # Track total size per sender
        
        total_sizes = {
            'pdf': 0,
            'word': 0,
            'image': 0
        }
        
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            headers = msg['payload']['headers']
            from_email = get_header(headers, 'From')
            subject = get_header(headers, 'Subject')
            date = get_header(headers, 'Date')
            
            if 'parts' in msg['payload']:
                for part in msg['payload']['parts']:
                    if part.get('filename'):
                        filename = part['filename']
                        attachment_type = get_attachment_type(filename)
                        attachment_id = part['body'].get('attachmentId')
                        
                        if attachment_id:
                            # Get attachment size
                            size = int(part['body'].get('size', 0))
                            
                            attachment = {
                                'id': attachment_id,
                                'filename': filename,
                                'from_email': from_email,
                                'subject': subject,
                                'date': date,
                                'message_id': message['id'],
                                'size': size,
                                'size_formatted': format_size(size)
                            }
                            
                            # Update sender statistics
                            sender = from_email.split('<')[0].strip()
                            sender_sizes[sender] += size
                            
                            if attachment_type == 'pdf':
                                total_sizes['pdf'] += size
                                pdf_attachments.append(attachment)
                            elif attachment_type == 'word':
                                total_sizes['word'] += size
                                word_attachments.append(attachment)
                            elif attachment_type == 'image':
                                total_sizes['image'] += size
                                attachment['preview_url'] = f'/get_attachment_preview/{attachment_id}'
                                image_attachments.append(attachment)
        
        # Get top 5 senders by total size
        top_senders = dict(sorted(sender_sizes.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:5])
        
        # Format total sizes
        formatted_sizes = {
            'pdf': format_size(total_sizes['pdf']),
            'word': format_size(total_sizes['word']),
            'image': format_size(total_sizes['image']),
            'total': format_size(sum(total_sizes.values()))
        }
        
        return (
            pdf_attachments, 
            word_attachments, 
            image_attachments, 
            formatted_sizes,
            {
                'senders': list(top_senders.keys()),
                'sizes': [format_size(size) for size in top_senders.values()],
                'raw_sizes': list(top_senders.values())
            }
        )
    except Exception as e:
        print(f"Error getting attachments: {e}")
        return [], [], [], {}, {'senders': [], 'sizes': [], 'raw_sizes': []}

def format_size(size):
    """Format file size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

@app.route('/attachments')
def show_attachments():
    """Display attachments page with statistics"""
    if 'credentials' not in session:
        return redirect('/')
    
    try:
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        
        pdf_attachments, word_attachments, image_attachments, sizes, sender_stats = get_attachments(service)
        
        return render_template(
            'attachments.html',
            pdf_attachments=pdf_attachments,
            word_attachments=word_attachments,
            image_attachments=image_attachments,
            sizes=sizes,
            sender_stats=sender_stats
        )
    except Exception as e:
        return f"Error fetching attachments: {str(e)}", 500

@app.route('/get_attachment/<attachment_id>')
def get_attachment(attachment_id):
    """Get attachment content"""
    if 'credentials' not in session:
        return {'error': 'Not authenticated'}, 401
    
    try:
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        
        # Get attachment content
        attachment = service.users().messages().attachments().get(
            userId='me',
            messageId=request.args.get('message_id'),
            id=attachment_id
        ).execute()
        
        # Decode attachment data
        file_data = base64.urlsafe_b64decode(attachment['data'])
        
        # Create a temporary file in the static directory
        filename = f"attachment_{attachment_id}_{int(time.time())}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(file_data)
        
        # Generate a URL for the file
        file_url = url_for('static', filename=filename, _external=True)
        
        return {
            'content_url': file_url,
            'from_email': request.args.get('from_email'),
            'subject': request.args.get('subject'),
            'date': request.args.get('date'),
            'filename': request.args.get('filename')
        }
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/get_attachment_preview/<attachment_id>')
def get_attachment_preview(attachment_id):
    """Get attachment preview (for images)"""
    if 'credentials' not in session:
        return {'error': 'Not authenticated'}, 401
    
    try:
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        
        # Get attachment content
        attachment = service.users().messages().attachments().get(
            userId='me',
            messageId=request.args.get('message_id'),
            id=attachment_id
        ).execute()
        
        # Decode attachment data
        file_data = base64.urlsafe_b64decode(attachment['data'])
        
        # Create a temporary file in the static directory
        filename = f"preview_{attachment_id}_{int(time.time())}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(file_data)
        
        # Generate a URL for the file
        file_url = url_for('static', filename=filename, _external=True)
        
        return redirect(file_url)
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/download_attachment/<attachment_id>')
def download_attachment(attachment_id):
    """Download attachment directly"""
    if 'credentials' not in session:
        return {'error': 'Not authenticated'}, 401
    
    try:
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        
        # Get attachment content
        attachment = service.users().messages().attachments().get(
            userId='me',
            messageId=request.args.get('message_id'),
            id=attachment_id
        ).execute()
        
        # Decode attachment data
        file_data = base64.urlsafe_b64decode(attachment['data'])
        
        # Create a temporary file in the static directory
        filename = request.args.get('filename', f"attachment_{attachment_id}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(file_data)
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return {'error': str(e)}, 500

def create_or_get_label(service, label_name):
    """Create a label if it doesn't exist, or get it if it does"""
    try:
        # Try to list all labels
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        
        # Check if our label exists
        for label in labels:
            if label['name'].lower() == label_name.lower():
                return label['id']
        
        # If we get here, we need to create the label
        label = {
            'name': label_name,
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show'
        }
        created_label = service.users().labels().create(userId='me', body=label).execute()
        return created_label['id']
    except Exception as e:
        print(f"Error creating/getting label: {e}")
        return None

def process_emails_for_subscriptions(service):
    """Process emails to identify and label subscription emails"""
    try:
        # Get the subscription label ID
        subscription_label_id = create_or_get_label(service, "Subscriptions")
        if not subscription_label_id:
            return
        
        # Get messages
        results = service.users().messages().list(userId='me', maxResults=100).execute()
        messages = results.get('messages', [])
        
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            # Check if email contains "unsubscribe" in headers or body
            headers = msg['payload']['headers']
            subject = get_header(headers, 'Subject')
            body = get_email_body(msg['payload'])
            
            if ('unsubscribe' in body.lower() or 
                'unsubscribe' in subject.lower() or 
                any('unsubscribe' in str(header).lower() for header in headers)):
                
                # Add subscription label
                service.users().messages().modify(
                    userId='me',
                    id=message['id'],
                    body={'addLabelIds': [subscription_label_id]}
                ).execute()
    except Exception as e:
        print(f"Error processing emails for subscriptions: {e}")

@app.route('/dashboard')
def show_dashboard():
    """Display email dashboard with charts and high priority emails"""
    if 'credentials' not in session:
        return redirect('/')
    
    try:
        credentials = flow.credentials
        service = build('gmail', 'v1', credentials=credentials)
        
        # Process emails for subscriptions first
        process_emails_for_subscriptions(service)
        
        # Get all labels
        labels_result = service.users().labels().list(userId='me').execute()
        labels = labels_result.get('labels', [])
        
        # Initialize counters
        tag_counts = {}
        subscription_sources = defaultdict(int)
        high_priority_emails = []
        
        # Find the Priority/high-priority label ID
        high_priority_label_id = None
        for label in labels:
            if label['name'] == 'Priority/high-priority':
                high_priority_label_id = label['id']
                break
        
        if high_priority_label_id:
            # Get high priority messages specifically
            high_priority_results = service.users().messages().list(
                userId='me',
                labelIds=[high_priority_label_id],
                maxResults=3
            ).execute()
            
            high_priority_messages = high_priority_results.get('messages', [])
            
            for message in high_priority_messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                high_priority_emails.append({
                    'subject': get_header(msg['payload']['headers'], 'Subject'),
                    'from': get_header(msg['payload']['headers'], 'From'),
                    'date': get_header(msg['payload']['headers'], 'Date'),
                    'summary': get_email_summary(msg['payload'])
                })
        
        # Get all messages for label counting
        results = service.users().messages().list(userId='me', maxResults=100).execute()
        messages = results.get('messages', [])
        
        # Create a set of user-generated categories to track
        user_categories = set()
        for label in labels:
            if label['type'] == 'user' and not (
                label['name'].startswith('Priority/') or 
                label['name'].startswith('Potential Deletion/') or
                label['name'] in ['Priority', 'Potential Deletion', 'Subscriptions']
            ):
                user_categories.add(label['name'])
        
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            # Count only user-generated category labels
            msg_labels = msg.get('labelIds', [])
            for label_id in msg_labels:
                label_name = next((l['name'] for l in labels if l['id'] == label_id), None)
                if label_name and label_name in user_categories:
                    tag_counts[label_name] = tag_counts.get(label_name, 0) + 1
            
            # Process subscription emails
            if any(l['name'] == 'Subscriptions' for l in labels if l['id'] in msg_labels):
                from_header = get_header(msg['payload']['headers'], 'From')
                sender = from_header.split('<')[0].strip()
                subscription_sources[sender] += 1
        
        # Sort subscription sources by count
        sorted_sources = dict(sorted(subscription_sources.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:10])  # Top 10 sources
        
        return render_template(
            'dashboard.html',
            tag_labels=list(tag_counts.keys()),
            tag_counts=list(tag_counts.values()),
            subscription_sources=list(sorted_sources.keys()),
            subscription_counts=list(sorted_sources.values()),
            high_priority_emails=high_priority_emails
        )
    except Exception as e:
        print(f"Error showing dashboard: {e}")
        return str(e), 500

@app.route('/voice')
def voice_assistant():
    if 'credentials' not in session:
        return redirect(url_for('authorize'))
    return render_template('voice.html')

@app.route('/voice/process', methods=['POST'])
def process_voice_command():
    if 'credentials' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    user_input = data.get('text', '').lower()
    
    try:
        # Load credentials from file
        with open('credentials.json', 'r') as f:
            client_config = json.load(f)
        
        # Initialize Gmail service with proper credentials
        creds = Credentials(
            token=session['credentials']['token'],
            refresh_token=session['credentials']['refresh_token'],
            token_uri=session['credentials']['token_uri'],
            client_id=client_config['installed']['client_id'],
            client_secret=client_config['installed']['client_secret'],
            scopes=session['credentials']['scopes']
        )
        
        service = build('gmail', 'v1', credentials=creds)
        
        # Process the command
        response = process_command(service, user_input)
        
        # Get the response text to speak
        response_text = response.get('response', '')
        
        # Speak the response in a separate thread to avoid blocking
        def speak_response():
            global engine
            try:
                # Create a new engine instance for each response
                current_engine = pyttsx3.init()
                current_engine.setProperty('rate', 150)
                current_engine.setProperty('volume', 0.9)
                
                # Speak the response
                current_engine.say(response_text)
                current_engine.runAndWait()
                
                # Clean up the current engine
                current_engine.stop()
                
            except Exception as e:
                print(f"Error in text-to-speech: {str(e)}")
                try:
                    # Try with a new engine instance
                    current_engine = pyttsx3.init()
                    current_engine.setProperty('rate', 150)
                    current_engine.setProperty('volume', 0.9)
                    current_engine.say(response_text)
                    current_engine.runAndWait()
                    current_engine.stop()
                except Exception as e2:
                    print(f"Failed to initialize text-to-speech engine: {str(e2)}")
        
        # Start the speech in a new thread
        threading.Thread(target=speak_response).start()
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in process_voice_command: {str(e)}")
        return jsonify({
            'error': str(e),
            'response': "Sorry, I encountered an error processing your request."
        }), 500

def process_command(service, command):
    # Initialize Gemini
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Add current command to history
    conversation_history.append({"role": "user", "content": command})
    
    # Create conversation context
    context = "\nPrevious conversation:\n"
    for msg in conversation_history[-5:]:  # Keep last 5 messages for context
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    
    # Create a prompt for Gemini to understand the command
    prompt = f"""
    You are a Gmail assistant. Here is the context of the conversation:
    {context}
    
    The user just said: "{command}"
    
    Please analyze this command and determine:
    1. What action is being requested (fetch, count, send, delete, create_label)
    2. What parameters are needed
    3. Whether confirmation is needed (only for delete actions)
    
    For fetch action, include:
    - limit: number of emails to fetch (default 5)
    - query: any specific search terms
    
    For count action, no parameters needed
    
    For send action, include:
    - to: recipient email (if not provided, set to None)
    - subject: email subject
    - body: email content
    
    For delete action, include:
    - message_id: ID of email to delete
    
    For create_label action, include:
    - label_name: name of the label to create
    
    If the command is about sending an email and all necessary information is provided,
    set confirmation_required to false.
    
    If the user is referring to a previous email or context, use that information to
    complete the current request.
    
    Respond in JSON format with the following structure:
    {{
        "action": "fetch|count|send|delete|create_label",
        "parameters": {{
            "param1": "value1",
            "param2": "value2"
        }},
        "confirmation_required": true|false,
        "response": "Your response to the user"
    }}
    
    Only respond with the JSON object, nothing else.
    """
    
    try:
        # Get Gemini's analysis
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean the response text to ensure it's valid JSON
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parse the JSON response
        analysis = json.loads(response_text)
        
        # For send action, check if we have all required parameters
        if analysis['action'] == 'send':
            required_params = ['to', 'subject', 'body']
            if all(param in analysis['parameters'] and analysis['parameters'][param] for param in required_params):
                analysis['confirmation_required'] = False
        
        # Execute the action if no confirmation is needed
        if not analysis.get('confirmation_required', True):
            result = execute_action(service, analysis)
            analysis['response'] = result
        
        # Add assistant's response to history
        conversation_history.append({"role": "assistant", "content": analysis['response']})
        
        return analysis
        
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
        error_response = "I'm having trouble understanding your request. Could you please try again?"
        conversation_history.append({"role": "assistant", "content": error_response})
        return {
            'action': 'error',
            'response': error_response,
            'confirmation_required': False
        }
    except Exception as e:
        print(f"Error in process_command: {str(e)}")
        error_response = f"Sorry, I encountered an error: {str(e)}"
        conversation_history.append({"role": "assistant", "content": error_response})
        return {
            'action': 'error',
            'response': error_response,
            'confirmation_required': False
        }

def execute_action(service, analysis):
    action = analysis.get('action')
    params = analysis.get('parameters', {})
    
    try:
        if action == 'fetch':
            # Fetch emails based on parameters
            limit = params.get('limit', 5)
            query = params.get('query', '')
            
            # Build the Gmail API query
            gmail_query = f'in:inbox {query}'
            results = service.users().messages().list(
                userId='me',
                maxResults=limit,
                q=gmail_query
            ).execute()
            
            messages = results.get('messages', [])
            if not messages:
                return "You don't have any emails matching your request."
            
            response = []
            for msg in messages:
                email = service.users().messages().get(
                    userId='me',
                    id=msg['id']
                ).execute()
                
                headers = email['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                from_email = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
                
                response.append(f"From: {from_email}\nSubject: {subject}\nDate: {date}\n")
            
            return "\n".join(response)
            
        elif action == 'count':
            # Count total emails
            results = service.users().messages().list(
                userId='me',
                maxResults=1
            ).execute()
            
            total_emails = results.get('resultSizeEstimate', 0)
            return f"You have approximately {total_emails} emails in your inbox."
            
        elif action == 'send':
            # Send email
            to_email = params.get('to')
            subject = params.get('subject', '')
            body = params.get('body', '')
            
            if not to_email:
                return "I need the recipient's email address to send the email."
            
            message = MIMEMultipart()
            message['to'] = to_email
            message['subject'] = subject
            message.attach(MIMEText(body))
            
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            return f"Email sent successfully to {to_email}"
            
        elif action == 'delete':
            # Delete email
            service.users().messages().delete(
                userId='me',
                id=params.get('message_id')
            ).execute()
            
            return "Email deleted successfully"
            
        elif action == 'create_label':
            # Create label
            label = {
                'name': params.get('label_name'),
                'messageListVisibility': 'show',
                'labelListVisibility': 'labelShow'
            }
            
            service.users().labels().create(
                userId='me',
                body=label
            ).execute()
            
            return f"Label '{params.get('label_name')}' created successfully"
            
        else:
            return "I'm not sure how to handle that command yet. Please try a different request."
            
    except Exception as e:
        print(f"Error in execute_action: {str(e)}")
        return f"Error executing action: {str(e)}"

if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Local development only
    app.run(host='localhost', port=5000, debug=True)