# Gmail Smart Labeler

A Flask web application that uses Google's Gemini AI to automatically categorize and label your Gmail messages.

## Features

- 🔐 Secure Google OAuth2 authentication
- 🤖 AI-powered email analysis using Gemini
- 🏷️ Automatic custom label creation and management
- 📧 Smart categorization of emails
- 🎨 Clean, modern user interface
- 🔄 Real-time label updates

## Prerequisites

- Python 3.8 or higher
- Google Cloud Project with:
  - Gmail API enabled
  - Gemini API enabled
- Google OAuth2 credentials (credentials.json)
- Gemini API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gmail-smart-labeler
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your credentials:
   - Place your Google OAuth2 credentials file as `credentials.json` in the project root
   - Create a `.env` file in the project root with the following content:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     FLASK_SECRET_KEY=your_secret_key_here
     ```

## Configuration

1. Environment Variables:
   - Create a `.env` file in the project root
   - Add your Gemini API key and Flask secret key
   - Never commit the `.env` file to version control

2. (Optional) Configure Flask settings in `app.py`:
```python
app.config.update({
    'SESSION_COOKIE_NAME': 'gmail-auth-session',
    'PERMANENT_SESSION_LIFETIME': 600,
    'SESSION_COOKIE_SECURE': False,
    'SESSION_COOKIE_SAMESITE': 'Lax'
})
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Sign in with your Google account
4. The application will:
   - Fetch your recent emails
   - Analyze them using Gemini AI
   - Create appropriate custom labels
   - Apply labels to your emails
   - Display the results in the interface

## How It Works

1. **Authentication**: Uses Google OAuth2 for secure access to your Gmail account
2. **Email Analysis**: 
   - Fetches email content and headers
   - Uses Gemini AI to analyze the content
   - Generates appropriate single-word topic labels
3. **Label Management**:
   - Creates new custom labels if needed
   - Applies labels to relevant emails
   - Maintains a clean interface showing only custom labels

## Security

- Uses secure OAuth2 authentication
- Implements session management
- Stores sensitive information in environment variables
- Follows Google's security best practices

## Limitations

- Only works with custom labels (ignores system labels)
- Limited to analyzing the 10 most recent emails
- Requires internet connection for API access

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gmail API
- Google Gemini AI
- Flask Framework 