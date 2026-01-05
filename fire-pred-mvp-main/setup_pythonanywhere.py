#!/usr/bin/env python3
"""
PythonAnywhere deployment script for SparkNet API
"""

import os
import sys

def create_deploy_files():
    """Create necessary files for PythonAnywhere deployment"""
    
    # Create requirements.txt for PythonAnywhere
    pa_requirements = """
fastapi==0.104.1
uvicorn[standard]==0.24.0
xgboost==1.7.6
pandas==2.1.4
numpy==1.24.3
dill==0.3.7
joblib==1.3.2
matplotlib==3.7.2
gunicorn==21.2.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(pa_requirements.strip())
    
    # Create wsgi.py for PythonAnywhere
    wsgi_content = """
import os
from main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
"""
    
    with open('wsgi.py', 'w') as f:
        f.write(wsgi_content.strip())
    
    print("✅ Created requirements.txt for PythonAnywhere")
    print("✅ Created wsgi.py for PythonAnywhere")
    print("\n📋 Next steps:")
    print("1. Sign up at https://www.pythonanywhere.com/")
    print("2. Get your API token from account settings")
    print("3. Run: pa create pythonanywhere.com --token YOUR_TOKEN")
    print("4. Run: pa upload /path/to/your/project")
    print("5. Your API will be live at: https://yourusername.pythonanywhere.com/api/explain")

if __name__ == "__main__":
    create_deploy_files()
