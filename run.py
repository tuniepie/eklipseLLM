import torch
import sys
import os


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Run the Streamlit app
from ui.streamlit_app import main

if __name__ == "__main__":
    main()