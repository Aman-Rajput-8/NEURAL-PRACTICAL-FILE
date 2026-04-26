import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Experiment 10: Implementation of Recurrent and Convolutional Neural Networks
# This script is built using PyTorch to ensure compatibility across all environments.

def run_rnn():
    print("\n--- RNN Visual Matrix Simulation (PyTorch) ---")
    text = "Neural Networks are powerful advanced tools"
    
    # Map the unique characters out of our string to index numbers (and vice versa)
    chars = sorted(list(set(text)))
    char_to_index = {char: i for i, char in enumerate(chars)}
    index_to_char = {i: char for i, char in enumerate(chars)}
    
    # We will look at chunks of 4 letters to predict the 5th letter
    seq_length = 4
    sequences, labels = [], []
    for i in range(len(text) - seq_length):
        seq = text[i:i + seq_length]
        label = text[i + seq_length]
        sequences.append([char_to_index[char] for char in seq])
        labels.append(char_to_index[label])
        
    X = torch.nn.functional.one_hot(torch.tensor(sequences), num_classes=len(chars)).float()
    y = torch.tensor(labels)

    # Define a simple basic RNN network
    class SimpleRNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(input_size=len(chars), hidden_size=8, batch_first=True) 
            self.fc = nn.Linear(8, len(chars))
            
        def forward(self, x):
            out, hidden = self.rnn(x)
            return self.fc(out[:, -1, :]), hidden

    model = SimpleRNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training RNN Model (50 Epochs)....")
    for epoch in range(50):
        optimizer.zero_grad()
        output, _ = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    start_seq = "Neur"
    print(f"\n1. Original Input Text Seed:")
    print(f"'{start_seq}'")
    
    with torch.no_grad():
        # Trace exactly 1 step visually to track matrices cleanly 
        x_input = [char_to_index.get(char, 0) for char in start_seq]
        print(f"\n2. Input string mapped to Dictionary Index Array:")
        print(x_input)
        
        x_tensor = torch.nn.functional.one_hot(torch.tensor([x_input]), num_classes=len(chars)).float()
        print(f"\n3. One-Hot Encoded Matrix representation of input text (Shape: {list(x_tensor.shape)}):")
        print(x_tensor.squeeze().detach().numpy().astype(int))

        # Pass through RNN layer explicitly
        rnn_out, hidden_state = model.rnn(x_tensor)
        print(f"\n4. RNN Hidden Layer Activation Matrix (Shape: {list(hidden_state.shape)}):")
        print(hidden_state.squeeze().detach().numpy().round(3))
        
        # Pass through Fully Connected Dense Layer explicitly
        dense_out = model.fc(rnn_out[:, -1, :])
        print(f"\n5. Dense Layer Output Probability Vectors (Shape: {list(dense_out.shape)}):")
        print(dense_out.squeeze().detach().numpy().round(2))
        
        # Make the final prediction based on the highest probability
        prediction = torch.softmax(dense_out, dim=1)
        predicted_idx = torch.argmax(prediction).item()
        next_char = index_to_char[predicted_idx]
        
        print(f"\n6. Logits Argmax executed! Predicted Index highest probability: {predicted_idx}")
        print(f"Next Character mapped: '{next_char}'\n")


def run_cnn():
    print("\n--- CNN Visual Matrix Simulation (PyTorch) ---")
    
    # Generate a tiny 6x6 mock image (a white square on a black background)
    mock_image = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
    mock_image[0, 0, 1:5, 1:5] = 1.0  

    print("\n1. Original Base Image Matrix (6x6 pixel grid):")
    print(mock_image.squeeze().numpy())

    # Map a standard 3x3 Edge-Detection convolution kernel
    edge_kernel = torch.tensor([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=torch.float32).view(1, 1, 3, 3)

    print("\n2. Executing Convolution... (Isolating grid edges)")
    conv_result = torch.nn.functional.conv2d(mock_image, edge_kernel, padding=1)
    print(conv_result.squeeze().detach().numpy())

    print("\n3. Executing ReLU Activation... (Destroying negative boundaries)")
    relu_result = torch.relu(conv_result)
    print(relu_result.squeeze().detach().numpy())

    print("\n4. Executing Max Pooling... (Dimensional layout scaling visually down to 3x3)")
    pool_result = torch.nn.functional.max_pool2d(relu_result, kernel_size=2, stride=2)
    print(pool_result.squeeze().detach().numpy())

    print("\n5. Executing Flatten... (Converting 3x3 grid into 1D Array)")
    flattened = torch.flatten(pool_result, start_dim=1)
    print(flattened.squeeze().detach().numpy())

    print("\n6. Executing Dense Output... (Yielding 16 Neurons)")
    dense_layer = nn.Linear(flattened.shape[1], 16)
    final_output = dense_layer(flattened)
    print(final_output.squeeze().detach().numpy().round(3))
    print()


if __name__ == "__main__":
    print("Available Models: 1. CNN  |  2. RNN")
    choice = input("Enter the model you want to simulate (CNN / RNN): ").strip().upper()
    
    if choice == 'RNN':
        run_rnn()
    elif choice == 'CNN':
        run_cnn()
    else:
        print("Invalid Selection.")
