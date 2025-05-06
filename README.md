***Classifying the Quality of Cardinality Estimation Using a Variational Quantum Classifier (VQC)***

This project uses a Variational Quantum Classifier (VQC) built with Qiskit to classify the error in cardinality estimation of database queries. The code processes query data from a CSV file, maps and scales features, builds quantum circuits (a feature map and three different ansatz circuits), trains the VQC, and generates a comparison of accuracies.

**Table of Contents**

* Installing Anaconda
* Setting Up the Environment
* Running the Code
* Project Structure
* Usage and Output
* Additional Notes

**Installing Anaconda**
* Download Anaconda:

    * Go to the Anaconda Distribution download page.
    * Download the Anaconda installer for Windows (choose the 64-bit version if your system supports it).
* Install Anaconda:
    * Run the downloaded installer.
    * Choose the default installation settings.
    * When prompted, you can check the box to add Anaconda to your PATH
* Verify Installation:
    * Open the Anaconda Prompt (you can find it in your Start Menu under Anaconda).
    * Type the following command and press Enter:

    ```console
    conda --version
    ```  
    You should see the version of conda printed.

**Setting Up the Environment**
* Open the Anaconda Prompt:
    * Launch the Anaconda Prompt from your Start Menu.      
* Create a New Environment:
    * Create a new conda environment named, for example, qml-env with Python 3.9 (or your preferred version):
     ```console
    conda create -n qml-env python=3.9
    ```  
* Activate the Environment:
    * Activate the newly created environment using this command
    ```console
    conda activate qml-env
    ```  
    * Your command prompt should now show (qml-env) at the beginning of the line.
* Install Required Packages:
    * With the environment activated, install the necessary Python packages using pip
    ```console
    pip install qiskit qiskit-aer qiskit-algorithms qiskit-machine-learning pandas numpy matplotlib scikit-learn
    ``` 
    * This command installs Qiskit (and its subpackages), along with other required libraries like pandas, numpy, matplotlib, and scikit-learn.

**Running the Code**
* Download the Project Code:
    * Ensure you have the project files (e.g., classification_of_cardinality_estimation.py and your CSV file cardErrorData.csv) in a directory on your computer.
* Navigate to the Project Directory:
    * In the Anaconda Prompt, change to the directory where your project files are located using the cd command. For example:
    ```console
    cd C:\Users\YourUsername\Path\To\Your\Project
    ``` 
* Run the Python Script:
    * With your environment activated and in the correct directory, run the script by typing:
    example:
    ```console
    python classification_of_cardinality_estimation.py
    ``` 
    * The code will load the CSV data, preprocess it, train the VQC classifiers using three different ansatz circuits, print accuracies, and generate graphs along with printing the circuit diagrams.