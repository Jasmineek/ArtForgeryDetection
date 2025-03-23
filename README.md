# Van Gogh Artwork Authenticator

A Flask web application that uses deep learning to detect whether a Van Gogh artwork is authentic or forged.

## Features

- User authentication system
- Drag-and-drop image upload
- Real-time artwork analysis
- Confidence score visualization
- Modern, responsive UI

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Jasmineek/ArtForgeryDetection.git
cd ArtForgeryDetection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and go to: http://127.0.0.1:5055

## Demo Credentials

- Username: admin, Password: admin123
- Username: user, Password: user123

## Project Structure

```
ArtForgeryDetection/
├── app.py                 # Main Flask application
├── templates/            # HTML templates
│   ├── login.html       # Login page
│   └── presentation.html # Main upload page
├── static/              # Static files
│   └── uploads/         # Uploaded images
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Technologies Used

- Flask
- TensorFlow
- Bootstrap 5
- Font Awesome
- Matplotlib

## License

This project is licensed under the MIT License - see the LICENSE file for details. 