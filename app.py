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

# --- Constituency Label Explanations ---
# Based on Penn Treebank tags, but can be customized
CONSTITUENCY_LABELS = {
    "S": "Simple declarative clause",
    "SBAR": "Clause introduced by a subordinating conjunction",
    "SBARQ": "Direct question introduced by a wh-word or wh-phrase",
    "SINV": "Inverted declarative sentence",
    "SQ": "Inverted yes/no question, or main clause of a wh-question",
    "NP": "Noun Phrase",
    "VP": "Verb Phrase",
    "PP": "Prepositional Phrase",
    "ADJP": "Adjective Phrase",
    "ADVP": "Adverb Phrase",
    "QP": "Quantifier Phrase (inside NP)",
    "WHNP": "Wh-noun Phrase",
    "WHPP": "Wh-prepositional Phrase",
    "WHADVP": "Wh-adverb Phrase",
    "PRN": "Parenthetical",
    "FRAG": "Fragment",
    "INTJ": "Interjection",
    "LST": "List marker",
    "UCP": "Unlike Coordinated Phrase",
    "CONJP": "Conjunction Phrase",
    "NX": "Used within certain complex NPs",
    "X": "Unknown, uncertain, or unbracketable",
    "ROOT": "Root of the tree (often implicit, added by some parsers)",
    # --- Common POS Tags (Penn Treebank Style) ---
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition or subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund or present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb",
    ".": "Punctuation, sentence end",
    ",": "Punctuation, comma",
    ":": "Punctuation, colon",
    "(": "Punctuation, open parenthesis",
    ")": "Punctuation, close parenthesis",
    "\"": "Punctuation, quotation mark",
    "`": "Punctuation, backtick",
    "#": "Punctuation, hash",
    "$": "Punctuation, dollar sign",
    "''": "Punctuation, closing quotation mark",
    "``": "Punctuation, opening quotation mark",
}
# --- End Constituency Label Explanations ---


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
    dependency_html_output = None
    constituency_parse_string = None # For benepar raw string output
    constituency_tree_json = None # For benepar JSON tree output
    dependency_explanations = None # Renamed for clarity
    constituency_explanations = None # For constituency labels
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
                    # --- Dependency Explanation Logic ---
                    unique_deps = sorted(list(set(token.dep_ for token in doc)))
                    dependency_explanations = {dep: spacy.explain(dep) for dep in unique_deps if spacy.explain(dep)}
                    # --- End Dependency Explanation Logic ---
                elif parse_type == 'constituency':
                    # Generate constituency parse string using benepar
                    # Ensure the benepar pipe has been added successfully
                    if 'benepar' in nlp.pipe_names:
                        sent = list(doc.sents)[0] # Get the first sentence
                        constituency_parse_string = sent._.parse_string
                        # --- Convert constituency string to NLTK Tree and then to JSON ---
                        try:
                            nltk_tree = nltk.Tree.fromstring(constituency_parse_string)
                            constituency_tree_json = tree_to_json(nltk_tree) # Convert to JSON for D3
                            # --- Constituency Explanation Logic ---
                            if nltk_tree:
                                unique_const_labels = get_labels_from_tree(nltk_tree)
                                constituency_explanations = {
                                    label: CONSTITUENCY_LABELS.get(label, "No description available.")
                                    for label in sorted(list(unique_const_labels))
                                    if label in CONSTITUENCY_LABELS # Only include known labels
                                }
                            # --- End Constituency Explanation Logic ---
                        except Exception as tree_e:
                            error_message = f"Error parsing constituency tree: {tree_e}"
                            print(f"Error parsing tree '{constituency_parse_string}': {tree_e}")
                            constituency_tree_json = None # Ensure it's None on error
                            constituency_explanations = None
                        # --- End Tree Conversion ---
                    else: # This is the correct 'else' for the 'if benepar in nlp.pipe_names'
                        error_message = "Constituency parsing component (benepar) not loaded correctly."
                        constituency_parse_string = "Error: benepar not available."
                        constituency_tree_json = None
                        constituency_explanations = None
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
                           dependency_explanations=dependency_explanations, # Pass CORRECT dependency explanations
                           constituency_explanations=constituency_explanations, # Pass constituency explanations
                           error=error_message,
                           input_sentence=sentence,
                           selected_parse_type=parse_type) # Pass selected type

# --- Function to extract unique labels from NLTK Tree ---
def get_labels_from_tree(tree):
    """Recursively extracts all unique node labels from an NLTK Tree."""
    labels = set()
    if not isinstance(tree, str): # Ignore leaf strings (words)
        labels.add(tree.label())
        for child in tree:
            labels.update(get_labels_from_tree(child))
    return labels
# --- End get_labels_from_tree function ---


# --- Function to convert NLTK Tree to JSON ---
def tree_to_json(tree):
    """Converts an NLTK Tree object to a JSON-serializable dictionary for D3."""
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
