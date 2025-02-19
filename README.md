# Cora GCN Node Classification

## Description
This project implements a Graph Convolutional Network (GCN) for node classification on the Cora citation dataset. In this dataset, each node represents a scientific paper and edges represent citation relationships between papers. Each paper comes with features (derived from the text) and a label indicating its research topic. The goal is to classify each node into its correct category by leveraging both its own features and the structure of the graph.

## Project Structure
- **main.py**  
  Contains the code to load the Cora dataset, define a simple 2-layer GCN model, train the model using node features and graph connectivity, evaluate its performance on validation and test sets, and save the trained model.
- **README.md**  
  Provides an overview of the project, installation instructions, usage details, and an explanation of how the model works.

## Installation
1. **Python Version:** 

    Use a compatible Python version (ideally between 3.7 and 3.9).
2. **Environment Setup:**  
    
    Create and activate a virtual environment.
3. **Required Packages:** 
    
    Install the required libraries, including PyTorch, PyTorch Geometric, and SciPy (plus any additional dependencies as per the PyTorch Geometric installation instructions).
    ```
    pip3 install -r requirements.txt
    ```

## Usage
- **Training:**  
  Run the `main.py` script to load the Cora dataset, train the GCN on the node classification task, evaluate its performance (printing accuracy on training, validation, and test nodes), and save the model.
  ```bash
  python3 main.py
  ```
- **Inference:**  
Run the `inference.py` script to load the saved model and run test data (or new data) on it. The script loads the model, runs the forward pass on the dataset, and prints out the test accuracy.
  ```bash
  python3 inference.py
  ```

## Key Concepts
- **Graph Convolutional Networks (GCNs):**  
  GCNs are neural networks designed for graph-structured data. They aggregate information from a node’s immediate neighbors (and, with multiple layers, from neighbors of neighbors) to produce richer node representations.
- **Node Classification:**  
  The model predicts a label for each node using its features and the information gathered from connected nodes. In the Cora dataset, this means assigning each paper a research topic.
- **Evaluation Metric:**  
  The test accuracy (e.g., 0.7820) represents the proportion of nodes in the test set that were correctly classified by the model.

## Evaluation and Purpose
The model’s accuracy indicates how well it can predict the correct research topic for unseen nodes based on both their content and their citation relationships. This approach is useful for understanding relational data in various applications, such as recommendation systems, social network analysis, and any domain where data is naturally connected in a graph structure.

## Conclusion
This project demonstrates the application of a GCN for node classification. It shows how combining node features with the graph's connectivity enables the model to make informed predictions, ultimately grouping similar nodes together. This serves as a foundation for more complex graph-based learning tasks and applications.
