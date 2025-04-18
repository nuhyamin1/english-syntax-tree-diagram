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
        # This case might happen for leaf nodes if not handled carefully
        return tree # Or handle as needed, maybe {'label': 'TOKEN', 'text': tree}

    node = {}
    node['label'] = tree.label()
    # Check if the node is a pre-terminal (POS tag) node
    is_preterminal = all(isinstance(child, str) for child in tree)
    if is_preterminal and len(tree) == 1:
        node['text'] = tree[0] # Assign the word as text
    else:
        # Internal node or terminal node with complex structure (shouldn't happen with standard benepar output)
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

        # Output display
        self.output_label = QLabel("Parse Output:")
        self.layout.addWidget(self.output_label)
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.layout.addWidget(self.output_display)

        # Web view for dependency parse visualization
        self.web_view = QWebEngineView()
        self.web_view.setVisible(False)
        self.layout.addWidget(self.web_view)

    def generate_constituency_parse(self):
        sentence = self.sentence_input.toPlainText().strip()
        if not sentence:
            self.output_display.setText("Please enter a sentence.")
            return

        try:
            # Process with SpaCy and Benepar
            if 'benepar' not in nlp.pipe_names:
                 self.output_display.setText("Error: Benepar component not loaded. Cannot generate constituency parse.")
                 return

            doc = nlp(sentence)
            if not list(doc.sents):
                 self.output_display.setText("Could not segment sentence.")
                 return

            # Assuming one sentence for simplicity in the GUI
            sent = list(doc.sents)[0]
            constituency_parse_string = sent._.parse_string

            if not constituency_parse_string:
                self.output_display.setText("No parse found by benepar.")
                return

            # --- Convert constituency string to NLTK Tree and then to JSON ---
            try:
                nltk_tree = Tree.fromstring(constituency_parse_string)
                constituency_tree_json = tree_to_json(nltk_tree)
                json_data_string = json.dumps(constituency_tree_json) # Convert dict to JSON string

                # --- Generate HTML for D3 Visualization ---
                html_content = self.generate_d3_html(json_data_string)
                self.web_view.setHtml(html_content)
                self.web_view.setVisible(True)
                self.output_display.setVisible(False)

            except Exception as tree_e:
                 self.output_display.setText(f"Error parsing/visualizing constituency tree: {tree_e}\n\nRaw parse:\n{constituency_parse_string}")
                 self.web_view.setVisible(False)
                 self.output_display.setVisible(True)
            # --- End Tree Conversion and Visualization ---

        except Exception as e:
            self.output_display.setText(f"Error generating constituency parse: {str(e)}")
            self.web_view.setVisible(False)
            self.output_display.setVisible(True)

    def generate_dependency_parse(self):
        sentence = self.sentence_input.toPlainText().strip()
        if not sentence:
            self.output_display.setText("Please enter a sentence.")
            return

        try:
            # Process with SpaCy
            doc = nlp(sentence)
            # Generate dependency parse visualization
            html = displacy.render(doc, style="dep", options={"compact": True})
            # Display in web view
            self.web_view.setHtml(html)
            self.web_view.setVisible(True)
            self.output_display.setVisible(False)
        except Exception as e:
            self.output_display.setText(f"Error generating dependency parse: {str(e)}")
            self.web_view.setVisible(False)
            self.output_display.setVisible(True)

    def generate_d3_html(self, json_data):
        # D3 rendering script extracted from index.html
        d3_script = """
            function renderD3Tree(data) {
                const container = d3.select("#constituency-tree-container");
                const svgElement = d3.select("#d3-tree-svg");
                svgElement.selectAll("*").remove(); // Clear previous tree

                if (!data || !container.node()) return;

                // --- Dynamic Width Calculation ---
                // Get container width, subtract padding/margins
                const availableWidth = container.node().getBoundingClientRect().width - 40; // e.g., 20px padding left/right

                // --- Tree Layout Setup ---
                const root = d3.hierarchy(data, d => d.children);
                const treeLayout = d3.tree();

                // --- Dynamic Height Calculation ---
                let maxDepth = 0;
                let nodesPerDepth = {};
                root.each(d => {
                    if (d.depth > maxDepth) maxDepth = d.depth;
                    nodesPerDepth[d.depth] = (nodesPerDepth[d.depth] || 0) + 1;
                });

                // Estimate height based on depth and node separation
                const nodeHeightSeparation = 80; // Vertical distance between levels
                const estimatedHeight = (maxDepth + 1) * nodeHeightSeparation;

                // --- Set Tree Size ---
                // Use availableWidth for horizontal spread, estimatedHeight for vertical
                treeLayout.size([availableWidth, estimatedHeight]);
                treeLayout(root); // Calculate node positions (d.x, d.y)

                // --- Adjust SVG Size ---
                // Add margins back to width and height for padding inside SVG
                svgElement.attr("width", availableWidth + 40)
                          .attr("height", estimatedHeight + 40); // Add bottom margin

                // --- Create SVG Group for Transformation ---
                // Translate the group to account for top/left margins
                const g = svgElement.append("g")
                                  .attr("transform", "translate(20, 40)"); // Left margin 20, Top margin 40

                // --- Draw Links (Edges) ---
                const link = g.selectAll(".link")
                    .data(root.links())
                    .enter().append("path")
                    .attr("class", "link")
                    .attr("d", d3.linkVertical() // Use vertical layout links
                        .x(d => d.x)
                        .y(d => d.y));

                // --- Draw Nodes (Groups containing circle and text) ---
                const node = g.selectAll(".node")
                    .data(root.descendants())
                    .enter().append("g")
                    .attr("class", d => "node" + (d.children ? " node--internal" : " node--leaf"))
                    .attr("transform", d => `translate(${d.x},${d.y})`); // Position node group

                // Add circle marker for each node
                node.append("circle")
                    .attr("r", 5);

                // Add Label (POS tag or Phrase label)
                node.append("text")
                    .attr("dy", "-0.8em") // Position above the node circle
                    .attr("text-anchor", "middle") // Center text horizontally
                    .attr("class", "label")
                    .text(d => d.data.label);

                // Add Text (Word/Token) - only for leaf nodes that have text
                node.filter(d => d.data.text)
                    .append("text")
                    .attr("dy", "1.8em") // Position below the node circle
                    .attr("text-anchor", "middle") // Center text horizontally
                    .attr("class", "text")
                    .text(d => d.data.text);
            }
        """

        # CSS styles extracted from index.html
        css_styles = """
            body { margin: 0; padding: 0; font-family: sans-serif; }
            #constituency-tree-container {
                padding: 20px;
                overflow: auto; /* Enable scrolling if tree is large */
                min-height: 300px; /* Ensure some height */
            }
            .node circle {
                fill: #fff;
                stroke: steelblue;
                stroke-width: 2px; /* Adjusted from 3px */
            }
            .node text {
                font: 11px sans-serif; /* Adjusted size */
            }
            .node .label {
                fill: #007bff; /* Blue label */
                font-weight: bold;
            }
            .node .text {
                fill: #28a745; /* Green text */
                font-style: italic;
            }
            .link {
                fill: none;
                stroke: #ccc;
                stroke-width: 1.5px; /* Adjusted from 2px */
            }
            svg {
                display: block; /* Prevent extra space below SVG */
            }
        """

        # Combine into a full HTML document
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
            <script>
                const treeData = {json_data};
                {d3_script}
                // Initial render
                if (treeData) {{
                    renderD3Tree(treeData);
                    // Optional: Re-render on window resize (might need adjustments in Qt context)
                    // window.addEventListener('resize', () => renderD3Tree(treeData));
                }}
            </script>
        </body>
        </html>
        '''
        return html


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SyntaxTreeApp()
    window.show()
    sys.exit(app.exec())
