import numpy as np
import re
import matplotlib.pyplot as plt
import ast

def draw_loss_graph(log_file: str):
    with open(log_file, 'r') as file:
        log_data = file.readlines()

    # Initialize lists to store epoch, training loss and validation loss
    epochs = []
    train_losses = []
    valid_losses = []

    # Regex patterns to match log lines
    train_loss_pattern = re.compile(r'Epoch (\d+): Train loss: ([\d.]+)')
    valid_loss_pattern = re.compile(r'Epoch (\d+): Train loss: ([\d.]+), Valid loss: ([\d.]+)')

    # Parse log data
    for line in log_data:
        train_match = train_loss_pattern.search(line)
        valid_match = valid_loss_pattern.search(line)
        
        if valid_match:
            epoch = int(valid_match.group(1))
            train_loss = float(valid_match.group(2))
            valid_loss = float(valid_match.group(3))
            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        elif train_match:
            epoch = int(train_match.group(1))
            train_loss = float(train_match.group(2))
            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(None)  # No validation loss for this epoch

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot([epoch for epoch, loss in zip(epochs, valid_losses) if loss is not None],
            [loss for loss in valid_losses if loss is not None],
            label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction + Hungarian Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def draw_ari_graph():
    ari_data = ["ari.txt", "ari_clevr.txt"]
    ari_values_list = []

    for file in ari_data:
        with open(file, 'r') as file:
            data_str = file.read()

        array = ast.literal_eval(data_str)
        ari_values_list.append(np.array(array))


    colors_list = ['#8A2BE2', '#4B0082']
    labels_list = ['SHAPES', 'CLEVR']

    c = ['#E75480', '#DA70D6', '#8A2BE2', '#4B0082', '#301934']


    num_models = len(ari_values_list)
    means = [np.mean(ari_values) for ari_values in ari_values_list]
    variances = [np.var(ari_values) for ari_values in ari_values_list]

    plt.bar(labels_list, means, yerr=variances, capsize=5, color=colors_list)
    plt.ylabel('Mean ARI')
    plt.title('Mean ARI and Variance for Multiple Models')

    # Display the plot
    plt.show()


draw_ari_graph()