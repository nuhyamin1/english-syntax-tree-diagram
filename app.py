from flask import Flask, render_template, request
import spacy
from spacy import displacy

# Load the spaCy model (ensure this happens only once)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    dependency_html_output = None
    explanations = None # Initialize explanations dictionary
    error_message = None
    sentence = ""

    if request.method == 'POST':
        sentence = request.form.get('sentence', '').strip()
        if sentence:
            try:
                # Process the sentence with spaCy
                doc = nlp(sentence)

                # --- Add Explanation Logic ---
                # Get unique dependency labels from the processed document
                unique_deps = sorted(list(set(token.dep_ for token in doc))) # Sort for consistent order
                # Create a dictionary mapping dep label to its explanation
                explanations = {dep: spacy.explain(dep) for dep in unique_deps if spacy.explain(dep)}
                # --- End Explanation Logic ---

                # Generate displaCy HTML (same as before)
                options = {
                    'compact': True,
                    'bg': '#fafafa',
                    'color': '#333333',
                    'font': 'Arial, sans-serif',
                    'distance': 120
                 }
                dependency_html_output = displacy.render(doc, style="dep", page=False, options=options)

            except Exception as e:
                error_message = f"An error occurred during processing: {e}"
                print(f"Error processing sentence '{sentence}': {e}")

        elif request.form:
             error_message = "Please enter a sentence."

    return render_template('index.html',
                           dependency_html_output=dependency_html_output,
                           explanations=explanations, # Pass the explanations dict
                           error=error_message,
                           input_sentence=sentence)

if __name__ == '__main__':
    app.run(debug=True)