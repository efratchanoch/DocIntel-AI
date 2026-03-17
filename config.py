from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
import os

# Load environment variables from .env file
load_dotenv()

# Access the 'API_KEY'
API_KEY = os.getenv("API_KEY")

if API_KEY is None:
    raise ValueError("API Key not found in .env file!")
else:
    print("Configuration Loaded Successfully")