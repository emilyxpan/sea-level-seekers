# ConvLSTM Model for Flood Prediction

## Overview

The **ConvLSTM model** is a deep learning architecture designed to handle spatiotemporal data, specifically to predict flooding events for 12 cities. It utilizes a sequence of **sea level anomaly (SLA) grids** over time as input to forecast the likelihood of flooding. 

### Key Components

1. **Initial Convolutional Layer**  
   Processes the input feature maps to extract spatial features.  

2. **Stack of Convolutional Layers**  
   Further refines spatial features for each time step in the input sequence.  

3. **ConvLSTM Cells**  
   - Combines spatial features with hidden and cell states from the previous time step.  
   - Retains temporal dependencies across the input sequence.  
   - Updates memory and hidden states for each time step.  

4. **Fully Connected Layer**  
   - Flattens the final hidden state.  
   - Produces a binary classification output for each city using a **sigmoid activation function**.

### Output
- The model outputs **probabilities** for the flooding status of each city.
- Predictions are in alphabetical order:
  - Index 0: Atlantic City  
  - Index 1: Baltimore  
  - ...  

## Running the Model

### Evaluation Command
To evaluate the model, use the following command:

```bash
python main.py load_model [path to model]
```

Note that this will load in a model in the main.py file. To use the model for a specific task, please add your own code using 
```bash
outputs = model(inputs)
```
where outputs are the 12 class predictions for the inputs. Note that city predictions occur in an alphabetical order (where index 0 of outputs is related to Atlantic City, index 1 to Baltimore, and so on).