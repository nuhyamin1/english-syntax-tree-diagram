# download_benepar.py
import nltk
import benepar
import sys

try:
    print("Attempting to download 'benepar_en3' using nltk...")
    # Ensure NLTK data path is configured if needed, though benepar.download might handle it.
    # nltk.data.path.append('/path/to/nltk_data') # Example if needed
    benepar.download('benepar_en3')
    print("Download command executed successfully.")
except Exception as e:
    print(f"An error occurred during download: {e}")
    sys.exit(1) # Exit with error code if download fails

print("Script finished.")