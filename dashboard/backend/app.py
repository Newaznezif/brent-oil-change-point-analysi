"""
Flask application for Brent Oil Change Point Analysis Dashboard
"""
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dashboard.backend.api import api_bp
from dashboard.backend.models import DataManager

app = Flask(__name__, 
           static_folder='../frontend/build',
           static_url_path='')
CORS(app)

# Register blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Initialize data manager
data_manager = DataManager()
app.config['DATA_MANAGER'] = data_manager

@app.route('/')
def serve_frontend():
    """Serve the frontend application"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Brent Oil Analysis Dashboard',
        'version': '1.0.0'
    })

@app.route('/api/dashboard/summary')
def dashboard_summary():
    """Get dashboard summary statistics"""
    try:
        summary = data_manager.get_dashboard_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Brent Oil Analysis Dashboard...")
    print(f"Data directory: {os.path.abspath('../../data')}")
    app.run(debug=True, host='0.0.0.0', port=5000)
