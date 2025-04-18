import sys
import spacy
from spacy import displacy
import benepar  # Import benepar
import nltk
from nltk import Tree # Import NLTK Tree for parsing the string
import json # To handle JSON data for D3
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QPushButton, QLabel)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt

# Load SpaCy model (ensure this happens only once)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy 'en_core_web_sm' model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load benepar model and add it to the pipeline
# Using similar logic as app.py for robustness
try:
    if spacy.__version__.startswith('2'):
        # Adjust based on your spacy version if needed
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        # For spaCy v3+
        if "benepar" not in nlp.pipe_names:
             nlp.add_pipe("benepar", config={"model": "benepar_en3"})
except ValueError as e:
    print(f"Benepar component issue: {e}. Attempting download...")
    try:
        import benepar.cli
        benepar.cli.download("benepar_en3")
        if "benepar" not in nlp.pipe_names:
             nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    except Exception as download_e:
        print(f"Failed to download or add benepar model: {download_e}")
        # Consider how the GUI should handle this - maybe disable the button?
        # For now, we'll let the parsing attempt fail later if benepar isn't loaded.
        pass

# --- Constituency Label Explanations (copied from app.py) ---
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


# --- Helper Functions (copied from app.py) ---
def get_labels_from_tree(tree):
    """Recursively extracts all unique node labels from an NLTK Tree."""
    labels = set()
    if not isinstance(tree, str): # Ignore leaf strings (words)
        labels.add(tree.label())
        for child in tree:
            labels.update(get_labels_from_tree(child))
    return labels

def tree_to_json(tree):
    """Converts an NLTK Tree object to a JSON-serializable dictionary for D3."""
    if isinstance(tree, str):
        return tree
    node = {}
    node['label'] = tree.label()
    is_preterminal = all(isinstance(child, str) for child in tree)
    if is_preterminal and len(tree) == 1:
        node['text'] = tree[0]
    else:
        node['children'] = [tree_to_json(child) for child in tree]
    return node
# --- End Helper Functions ---


class SyntaxTreeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("English Syntax Tree Generator")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Sentence input
        self.input_label = QLabel("Enter an English sentence:")
        self.layout.addWidget(self.input_label)
        self.sentence_input = QTextEdit()
        self.sentence_input.setFixedHeight(50)
        self.layout.addWidget(self.sentence_input)

        # Buttons for parsing
        self.constituency_btn = QPushButton("Generate Constituency Parse")
        self.constituency_btn.clicked.connect(self.generate_constituency_parse)
        self.layout.addWidget(self.constituency_btn)

        self.dependency_btn = QPushButton("Generate Dependency Parse")
        self.dependency_btn.clicked.connect(self.generate_dependency_parse)
        self.layout.addWidget(self.dependency_btn)

        # Output display (kept for errors)
        self.output_label = QLabel("Parse Output:")
        self.layout.addWidget(self.output_label)
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        # Hide initially, show only on error or if no web view content
        self.output_display.setVisible(False)
        self.layout.addWidget(self.output_display)

        # Web view for visualizations
        self.web_view = QWebEngineView()
        self.web_view.setVisible(False) # Start hidden
        self.layout.addWidget(self.web_view)

    def show_error(self, message):
        """Helper to display errors in the text area."""
        self.output_display.setText(message)
        self.output_display.setVisible(True)
        self.web_view.setVisible(False)

    def generate_constituency_parse(self):
        sentence = self.sentence_input.toPlainText().strip()
        if not sentence:
            self.show_error("Please enter a sentence.")
            return

        try:
            # Process with SpaCy and Benepar
            if 'benepar' not in nlp.pipe_names:
                 self.show_error("Error: Benepar component not loaded. Cannot generate constituency parse.")
                 return

            doc = nlp(sentence)
            if not list(doc.sents):
                 self.show_error("Could not segment sentence.")
                 return

            sent = list(doc.sents)[0]
            constituency_parse_string = sent._.parse_string

            if not constituency_parse_string:
                self.show_error("No parse found by benepar.")
                return

            # --- Convert constituency string to NLTK Tree and then to JSON ---
            try:
                nltk_tree = Tree.fromstring(constituency_parse_string)
                constituency_tree_json = tree_to_json(nltk_tree)
                json_data_string = json.dumps(constituency_tree_json)

                # --- Get Constituency Explanations ---
                unique_const_labels = get_labels_from_tree(nltk_tree)
                constituency_explanations = {
                    label: CONSTITUENCY_LABELS.get(label, "No description available.")
                    for label in sorted(list(unique_const_labels))
                    if label in CONSTITUENCY_LABELS
                }

                # --- Generate HTML for D3 Visualization with Legend ---
                html_content = self.generate_constituency_html(json_data_string, constituency_explanations)
                self.web_view.setHtml(html_content)
                self.web_view.setVisible(True)
                self.output_display.setVisible(False)

            except Exception as tree_e:
                 self.show_error(f"Error parsing/visualizing constituency tree: {tree_e}\n\nRaw parse:\n{constituency_parse_string}")

        except Exception as e:
            self.show_error(f"Error generating constituency parse: {str(e)}")

    def generate_dependency_parse(self):
        sentence = self.sentence_input.toPlainText().strip()
        if not sentence:
            self.show_error("Please enter a sentence.")
            return

        try:
            # Process with SpaCy
            doc = nlp(sentence)

            # --- Get Dependency Explanations ---
            unique_deps = sorted(list(set(token.dep_ for token in doc)))
            dependency_explanations = {dep: spacy.explain(dep) for dep in unique_deps if spacy.explain(dep)}

            # Generate dependency parse visualization SVG string
            # Use page=False to get only the SVG part
            options = {"compact": True, "bg": "#ffffff", "color": "#000000", "font": "Arial"}
            svg = displacy.render(doc, style="dep", options=options, page=False)

            # --- Generate HTML for Dependency Visualization with Legend ---
            html_content = self.generate_dependency_html(svg, dependency_explanations)
            self.web_view.setHtml(html_content)
            self.web_view.setVisible(True)
            self.output_display.setVisible(False)

        except Exception as e:
            self.show_error(f"Error generating dependency parse: {str(e)}")

    def generate_legend_html(self, explanations):
        """Generates the HTML list for a legend."""
        if not explanations:
            return ""
        items_html = "".join(f'<li><strong>{tag}:</strong> {desc}</li>'
                             for tag, desc in explanations.items())
        return f'''
            <div class="explanations-list">
                <h3>Legend</h3>
                <ul>{items_html}</ul>
            </div>
        '''

    def get_legend_css(self):
        """Returns the CSS for the legend."""
        return """
            .explanations-list {
                margin-top: 25px;
                padding: 15px;
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                font-size: 0.9em;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .explanations-list h3 {
                margin-top: 0;
                margin-bottom: 10px;
                color: #343a40;
                font-size: 1.1em;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 5px;
            }
            .explanations-list ul {
                list-style-type: none;
                padding-left: 0;
                margin: 0;
                max-height: 200px; /* Limit height and make scrollable */
                overflow-y: auto;
            }
            .explanations-list li {
                margin-bottom: 6px;
                padding: 4px 0;
            }
            .explanations-list strong {
                display: inline-block;
                min-width: 50px;
                font-weight: bold;
                margin-right: 8px;
                color: #495057;
            }
        """

    def generate_constituency_html(self, json_data, explanations):
        """Generates HTML for D3 Constituency Tree + Legend."""
        d3_script = """
            function renderD3Tree(data) {
                const container = d3.select("#constituency-tree-container");
                const svgElement = d3.select("#d3-tree-svg");
                svgElement.selectAll("*").remove(); // Clear previous tree

                if (!data || !container.node()) return;
                const availableWidth = container.node().getBoundingClientRect().width - 40;
                const root = d3.hierarchy(data, d => d.children);
                const treeLayout = d3.tree();
                let maxDepth = 0;
                root.each(d => { if (d.depth > maxDepth) maxDepth = d.depth; });
                const nodeHeightSeparation = 80;
                const estimatedHeight = (maxDepth + 1) * nodeHeightSeparation;
                treeLayout.size([availableWidth, estimatedHeight]);
                treeLayout(root);
                svgElement.attr("width", availableWidth + 40).attr("height", estimatedHeight + 40);
                const g = svgElement.append("g").attr("transform", "translate(20, 40)");
                g.selectAll(".link").data(root.links()).enter().append("path")
                    .attr("class", "link")
                    .attr("d", d3.linkVertical().x(d => d.x).y(d => d.y));
                const node = g.selectAll(".node").data(root.descendants()).enter().append("g")
                    .attr("class", d => "node" + (d.children ? " node--internal" : " node--leaf"))
                    .attr("transform", d => `translate(${d.x},${d.y})`);
                node.append("circle").attr("r", 5);
                node.append("text").attr("dy", "-0.8em").attr("text-anchor", "middle").attr("class", "label").text(d => d.data.label);
                node.filter(d => d.data.text).append("text").attr("dy", "1.8em").attr("text-anchor", "middle").attr("class", "text").text(d => d.data.text);
            }
        """
        css_styles = """
            body { margin: 10px; padding: 0; font-family: sans-serif; background-color: #f0f0f0; }
            #constituency-tree-container {
                padding: 20px;
                overflow: auto;
                min-height: 300px;
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-bottom: 15px; /* Space before legend */
            }
            .node circle { fill: #fff; stroke: steelblue; stroke-width: 2px; }
            .node text { font: 11px sans-serif; }
            .node .label { fill: #007bff; font-weight: bold; }
            .node .text { fill: #28a745; font-style: italic; }
            .link { fill: none; stroke: #ccc; stroke-width: 1.5px; }
            svg { display: block; }
        """ + self.get_legend_css() # Add legend CSS

        legend_html = self.generate_legend_html(explanations)

        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>{css_styles}</style>
        </head>
        <body>
            <div id="constituency-tree-container">
                <svg id="d3-tree-svg"></svg>
            </div>
            {legend_html}
            <script>
                const treeData = {json_data};
                {d3_script}
                if (treeData) {{ renderD3Tree(treeData); }}
            </script>
        </body>
        </html>
        '''
        return html

    def generate_dependency_html(self, svg_content, explanations):
        """Generates HTML for Dependency SVG + Legend."""
        css_styles = """
            body { margin: 10px; padding: 0; font-family: sans-serif; background-color: #f0f0f0; }
            .displacy-container {
                padding: 20px;
                overflow: auto;
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-bottom: 15px; /* Space before legend */
            }
            /* Add specific styles for displacy SVG if needed */
            .displacy-container svg {{ display: block; }}
        """ + self.get_legend_css() # Add legend CSS

        legend_html = self.generate_legend_html(explanations)

        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>{css_styles}</style>
        </head>
        <body>
            <div class="displacy-container">
                {svg_content}
            </div>
            {legend_html}
        </body>
        </html>
        '''
        return html


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # --- Apply Stylesheet ---
    stylesheet = """
        QMainWindow {
            background-color: #f0f0f0; /* Light grey background */
        }
        QWidget#central_widget { /* Target the central widget specifically */
            background-color: #f0f0f0;
            padding: 15px;
        }
        QLabel {
            font-size: 14px;
            color: #333;
            margin-bottom: 5px;
        }
        QTextEdit {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
            background-color: white;
        }
        QTextEdit#output_display { /* Specific style for read-only output */
             background-color: #e9e9e9;
        }
        QPushButton {
            background-color: #007bff; /* Blue background */
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 14px;
            margin-top: 10px; /* Add space above buttons */
        }
        QPushButton:hover {
            background-color: #0056b3; /* Darker blue on hover */
        }
        QPushButton:pressed {
            background-color: #004085; /* Even darker blue when pressed */
        }
        QWebEngineView {
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-top: 10px; /* Add space above web view */
        }
    """
    app.setStyleSheet(stylesheet)
    # --- End Stylesheet ---

    window = SyntaxTreeApp()
    # Set object name for the central widget to apply specific styles
    window.central_widget.setObjectName("central_widget")
    window.output_display.setObjectName("output_display") # Name the output display

    window.show()
    sys.exit(app.exec())
