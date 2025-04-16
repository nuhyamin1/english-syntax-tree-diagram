# app.py
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

# --- New Function for Bracketed Parse ---
def build_bracketed_string(token):
    """
    Recursively builds a LISP-style bracketed string for a token,
    including its POS tag, text, and children labeled with their dependency relations.
    Example output format: (POS Text (DepLabel Child1) (DepLabel Child2) ...)
    """
    # Sort children by their position in the sentence for readability
    children = sorted([child for child in token.children], key=lambda x: x.i)

    # Base case: Leaf node (no children)
    if not children:
        # Format: (POS Text)
        return f"({token.pos_} {token.orth_})"
    # Recursive case: Node with children
    else:
        child_strings = []
        for child in children:
            # Recursively get the child's structure
            child_structure = build_bracketed_string(child)
            # Wrap the child's structure with its dependency label relative to the current token
            # Format: (DepLabel ChildStructure)
            child_strings.append(f"({child.dep_} {child_structure})")

        # Combine the current token's info with its children's structures
        # Format: (POS Text ChildString1 ChildString2 ...)
        return f"({token.pos_} {token.orth_} {' '.join(child_strings)})"
# --- End New Function ---

@app.route('/', methods=['GET', 'POST'])
def index():
    dependency_html_output = None
    explanations = None # Initialize explanations dictionary
    error_message = None
    sentence = ""
    bracketed_parse = None # Initialize bracketed parse string

    if request.method == 'POST':
        sentence = request.form.get('sentence', '').strip()
        if sentence:
            try:
                # Process the sentence with spaCy
                doc = nlp(sentence)

                # --- Add Explanation Logic --- (Same as before)
                unique_deps = sorted(list(set(token.dep_ for token in doc)))
                explanations = {dep: spacy.explain(dep) for dep in unique_deps if spacy.explain(dep)}
                # --- End Explanation Logic ---

                # --- Generate Bracketed Parse String using the new function ---
                roots = [token for token in doc if token.head == token]
                if roots:
                    # Assuming a single root for most well-formed sentences
                    # Build the structure starting from the root
                    root_structure = build_bracketed_string(roots[0])
                    # Wrap the entire structure in a (ROOT ...) tag
                    bracketed_parse = f"(ROOT {root_structure})"
                    # Optional: Handle multiple roots if necessary (less common)
                    # if len(roots) > 1:
                    #    root_structures = [build_bracketed_string(r) for r in roots]
                    #    bracketed_parse = f"(MULTI_ROOT {' '.join(root_structures)})"
                else:
                     bracketed_parse = "(NO_ROOT_FOUND)" # Should generally not happen

                # --- End Bracketed Parse Generation ---


                # Generate displacy HTML (same as before)
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
                           input_sentence=sentence,
                           bracketed_parse=bracketed_parse) # Pass the new bracketed parse

if __name__ == '__main__':
    app.run(debug=True)