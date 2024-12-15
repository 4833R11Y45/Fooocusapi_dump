# launch.py
import os
import sys
from args_manager import args

# Disable WebUI
args.nowebui = True

# Set other necessary arguments
args.port = 7865
args.listen = "0.0.0.0"

# Import and run the main application
from ufurt_api import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host=args.listen, port=args.port)