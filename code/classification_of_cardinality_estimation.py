#!/usr/bin/env python
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import io

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC

# scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(file_path):
    """
    Load data from a CSV file and print a preview.
    """
    df = pd.read_csv(file_path)
    print("Data preview:")
    print(df.head())
    return df


def map_table_names(df):
    """
    Extract unique table names from the 'tables' column,
    map them to integers, and add a 'table_list' column.
    """
    table_names = set()

    def extract_tables(s):
        names = s.split(';')
        for name in names:
            table_names.add(name.strip())
        return names

    df['table_list'] = df['tables'].apply(extract_tables)
    table_to_int = {name: idx for idx, name in enumerate(sorted(table_names))}
    print("\nMapping of table names to integers:")
    print(table_to_int)
    return table_to_int, df


def parse_selectivities(df):
    """
    Convert the 'selectivities' column strings to lists of floats.
    """
    def parse_selectivities_str(s):
        return [float(x.strip()) for x in s.split(';')]
    df['sel_list'] = df['selectivities'].apply(parse_selectivities_str)
    return df


def preprocess_data(df, table_to_int, num_classes=4):
    """
    Preprocess the data by:
      - Determining the maximum number of tables (qubits)
      - Building feature vectors (padding and scaling)
      - Binning the error column into classes.
    Returns:
      X: feature matrix
      y: label vector
      n_qubits: number of qubits (max tables)
    """
    max_tables = df['table_list'].apply(len).max()
    print("\nMaximum number of tables in any query:", max_tables)
    n_qubits = max_tables

    # Use the maximum table integer for scaling.
    max_table_int = max(table_to_int.values()) if table_to_int else 1
    denom = max_table_int if max_table_int > 0 else 1

    def process_query(table_list, sel_list):
        # Convert table names to integers
        int_list = [table_to_int[name.strip()] for name in table_list]
        # Pad lists to length n_qubits
        padded_tables = int_list + [0] * (n_qubits - len(int_list))
        padded_sels = sel_list + [0.0] * (n_qubits - len(sel_list))
        # Scale values to [0, π]
        scaled_tables = [ (t / denom) * math.pi for t in padded_tables ]
        scaled_sels = [ s * math.pi for s in padded_sels ]
        # Build feature vector (concatenating the two values per qubit)
        feature_vector = []
        for t, s in zip(scaled_tables, scaled_sels):
            feature_vector.extend([t, s])
        return np.array(feature_vector)

    X = df.apply(lambda row: process_query(row['table_list'], row['sel_list']), axis=1).tolist()
    X = np.array(X)
    print("\nFeature matrix shape:", X.shape)

    # Bin the error column into classes.
    df['class'] = pd.qcut(df['error'], q=num_classes, labels=False)
    y = df['class'].values.astype(int)
    print("Labels shape:", y.shape)
    print("Unique classes:", np.unique(y))
    return X, y, n_qubits


def create_feature_map(n_qubits):
    """
    Create a quantum feature map that encodes 2 features per qubit.
    """
    feature_dim = 2 * n_qubits
    x = ParameterVector('x', feature_dim)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.rx(x[2 * i], i)
        qc.rz(x[2 * i + 1], i)
    return qc


def ansatz1(n_qubits):
    """
    Ansatz 1: Basic ansatz with Ry rotations and a chain of CNOTs.
    """
    num_params = n_qubits
    params = ParameterVector('theta', num_params)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(params[i], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def ansatz2(n_qubits):
    """
    Ansatz 2: Rx and Rz rotations on each qubit and all-to-all CNOT entanglement.
    """
    num_params = 2 * n_qubits
    params = ParameterVector('theta', num_params)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.rx(params[i], i)
        qc.rz(params[i + n_qubits], i)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qc.cx(i, j)
    return qc


def ansatz3(n_qubits):
    """
    Ansatz 3: Layered structure with Ry and Rz rotations and a ring of CZ gates.
    """
    num_params = n_qubits
    params = ParameterVector('theta', num_params)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(params[i], i)
        qc.rz(params[i], i)
    for i in range(n_qubits):
        qc.cz(i, (i + 1) % n_qubits)
    return qc


def train_classifiers(X, y, n_qubits, feature_map, optimizer, backend):
    """
    Train a VQC classifier for each ansatz.
    Displays the quantum feature map and ansatz circuit during training.
    Returns a dictionary with the accuracy results.
    """
    ansatz_functions = {
        'Ansatz 1': ansatz1,
        'Ansatz 2': ansatz2,
        'Ansatz 3': ansatz3
    }
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for name, ansatz_func in ansatz_functions.items():
        print("\n===================================")
        print("Training VQC using", name)
        
        # Build the ansatz circuit.
        ansatz_circuit = ansatz_func(n_qubits)
        
        # Create the VQC classifier.
        classifier = VQC(optimizer=optimizer,
                         feature_map=feature_map,
                         ansatz=ansatz_circuit)
        if hasattr(classifier, 'set_backend'):
            classifier.set_backend(backend)
        
        # Train the classifier.
        classifier.fit(X_train, y_train)
        
        # Evaluate the classifier.
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy using {name}: {acc:.3f} ({acc*100:.1f}%)")
        results[name] = acc
    return results


def generate_accuracy_graph(results):
    """
    Generate and display a bar chart comparing the accuracies.
    """
    ansatz_names = list(results.keys())
    accuracies = [results[name] for name in ansatz_names]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(ansatz_names, accuracies, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Comparison of VQC Ansatz Accuracies')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f'{yval:.3f}\n({yval*100:.1f}%)',
                 ha='center', va='bottom')
    plt.show()


def print_all_circuits(n_qubits, feature_map):
    """
    Print the quantum feature map and the circuit diagrams for each ansatz.
    """
    print("\nFinal Quantum Feature Map Circuit:")
    print(feature_map.draw(output='text'))
    
    for name, ansatz_func in {'Ansatz 1': ansatz1, 'Ansatz 2': ansatz2, 'Ansatz 3': ansatz3}.items():
        print(f"\nFinal Circuit Diagram for {name}:")
        circuit_diagram = ansatz_func(n_qubits)
        print(circuit_diagram.draw(output='text'))


def main():
    # Load data from CSV file
    file_path = "cardErrorData.csv" 
    df = load_data(file_path)
    
    # Map table names to integers
    table_to_int, df = map_table_names(df)
    df = parse_selectivities(df)
    
    # Preprocess data to create feature matrix X and labels y.
    X, y, n_qubits = preprocess_data(df, table_to_int, num_classes=4)
    
    # Create quantum feature map
    feature_map = create_feature_map(n_qubits)
    
    # Set up optimizer and backend.
    optimizer = COBYLA(maxiter=100)
    backend = Aer.get_backend('aer_simulator_statevector')
    
    # Train classifiers for each ansatz and collect results.
    results = train_classifiers(X, y, n_qubits, feature_map, optimizer, backend)
    
    print("\nSummary of accuracies:")
    for name, acc in results.items():
        print(f"{name}: {acc:.3f} ({acc*100:.1f}%)")
    
    # Draw all circuits used from training after printing the the accurecies
    print_all_circuits(n_qubits, feature_map)

    # Generate a graph comparing the accuracies
    generate_accuracy_graph(results)
    


if __name__ == "__main__":
    main()
