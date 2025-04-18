# app.py
from flask import Flask, render_template, request, jsonify
import spacy
from spacy import displacy
import benepar  # Import benepar
import nltk # Import NLTK for tree parsing

# Load the spaCy model (ensure this happens only once)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy 'en_core_web_sm' model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load benepar model and add it to the pipeline
try:
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        # For spaCy v3, ensure benepar is added correctly
        # Check benepar docs for the latest recommended way for spaCy v3+
        # This might involve specifying the component name explicitly
        if "benepar" not in nlp.pipe_names:
             nlp.add_pipe("benepar", config={"model": "benepar_en3"})
except ValueError as e:
    # Handle cases where the component might already be added or model not found
    print(f"Benepar component issue: {e}")
    # Attempt to download benepar model if not found (requires user confirmation ideally)
    # Consider adding more robust error handling or instructions here
    try:
        print("Attempting to download 'benepar_en3' model...")
        import benepar.cli
        benepar.cli.download("benepar_en3")
        # Retry adding the pipe after download
        if "benepar" not in nlp.pipe_names:
             nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    except Exception as download_e:
        print(f"Failed to download or add benepar model: {download_e}")
        # Decide how to proceed: maybe disable constituency parsing? Exit?
        pass # Or raise an error

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
    constituency_parse_string = None # For benepar raw string output
    constituency_tree_json = None # For benepar JSON tree output
    explanations = None # Initialize explanations dictionary
    error_message = None
    sentence = ""
    parse_type = 'dependency' # Default parse type

    if request.method == 'POST':
        sentence = request.form.get('sentence', '').strip()
        parse_type = request.form.get('parse_type', 'dependency') # Get selected parse type

        if sentence:
            try:
                # Process the sentence with spaCy
                doc = nlp(sentence)

                # --- Add Explanation Logic --- (Same as before)
                unique_deps = sorted(list(set(token.dep_ for token in doc)))
                explanations = {dep: spacy.explain(dep) for dep in unique_deps if spacy.explain(dep)}
                # --- End Explanation Logic ---

                # --- Generate Output based on Parse Type ---
                if parse_type == 'dependency':
                    # Generate displacy HTML for dependency parse
                    options = {
                        'compact': True,
                        'bg': '#fafafa',
                        'color': '#333333',
                        'font': 'Arial, sans-serif',
                        'distance': 120
                    }
                    dependency_html_output = displacy.render(doc, style="dep", page=False, options=options)
                elif parse_type == 'constituency':
                    # Generate constituency parse string using benepar
                    # Ensure the benepar pipe has been added successfully
                    if 'benepar' in nlp.pipe_names:
                        sent = list(doc.sents)[0] # Get the first sentence
                        constituency_parse_string = sent._.parse_string
                        # --- Convert constituency string to NLTK Tree and then to JSON ---
                        try:
                            nltk_tree = nltk.Tree.fromstring(constituency_parse_string)
                            constituency_tree_json = tree_to_json(nltk_tree)
                        except Exception as tree_e:
                            error_message = f"Error parsing constituency tree: {tree_e}"
                            print(f"Error parsing tree '{constituency_parse_string}': {tree_e}")
                            constituency_tree_json = None # Ensure it's None on error
                        # --- End Tree Conversion ---
                    else:
                        error_message = "Constituency parsing component (benepar) not loaded correctly."
                        constituency_parse_string = "Error: benepar not available."
                        constituency_tree_json = None
                # --- End Output Generation ---

            except Exception as e:
                error_message = f"An error occurred during processing: {e}"
                print(f"Error processing sentence '{sentence}': {e}")

        elif request.form:
             error_message = "Please enter a sentence."

    return render_template('index.html',
                           dependency_html_output=dependency_html_output,
                           constituency_parse_string=constituency_parse_string, # Pass raw string
                           constituency_tree_json=constituency_tree_json, # Pass JSON tree
                           explanations=explanations,
                           error=error_message,
                           input_sentence=sentence,
                           selected_parse_type=parse_type) # Pass selected type

# --- Function to convert NLTK Tree to JSON --- 
def tree_to_json(tree):
    """Converts an NLTK Tree object to a JSON-serializable dictionary."""
    if isinstance(tree, str):
        # This case might happen for leaf nodes if not handled carefully
        # Depending on how fromstring parses, might need adjustment
        return tree # Or handle as needed, maybe {'label': 'TOKEN', 'text': tree}

    node = {}
    node['label'] = tree.label()
    if len(tree) == 1 and isinstance(tree[0], str):
        # Leaf node: (POS Text)
        node['text'] = tree[0]
    else:
        # Internal node: (Label Child1 Child2 ...)
        node['children'] = [tree_to_json(child) for child in tree]
    return node
# --- End NLTK Tree to JSON function ---

if __name__ == '__main__':
    app.run(debug=True)