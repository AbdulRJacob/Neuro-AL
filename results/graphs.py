import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def draw_loss_graph(log_file: str):
    with open(log_file, 'r') as file:
        log_data = file.readlines()

    epochs = []
    train_losses = []
    valid_losses = []

    # Regex patterns to match log lines
    train_loss_pattern = re.compile(r'Epoch (\d+): Train loss: ([\d.]+)')
    valid_loss_pattern = re.compile(r'Epoch (\d+): Train loss: ([\d.]+), Valid loss: ([\d.]+)')

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
        with open(f"results/neuro_results/{file}", 'r') as file:
            data_str = file.read()

        array = ast.literal_eval(data_str)
        ari_values_list.append(np.array(array))


    colors_list = ['#8A2BE2', '#4B0082']
    labels_list = ['SHAPES', 'CLEVR']


    means = [np.mean(ari_values) for ari_values in ari_values_list]
    variances = [np.var(ari_values) for ari_values in ari_values_list]

    plt.bar(labels_list, means, yerr=variances, capsize=5, color=colors_list)
    plt.ylabel('Mean ARI')
    plt.title('Adjusted Rand Index (ARI)')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.savefig("ari_plot.png")


def plot_metrics(file_path):
    datasets = {}
    current_dataset = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Dataset:"):
                current_dataset = line.split(":")[1].strip()
                datasets[current_dataset] = {}
            elif current_dataset:
                match = re.match(r'(\w+) - Precision: ([\d.]+), Recall: ([\d.]+), Accuracy: ([\d.]+), F1 Score: ([\d.]+)', line)
                if match:
                    metric, precision, recall, accuracy, f1_score = match.groups()
                    datasets[current_dataset][metric] = {
                        'Precision': float(precision),
                        'Recall': float(recall),
                        'Accuracy': float(accuracy),
                        'F1 Score': float(f1_score)
                    }

    metrics = ['Precision', 'Recall', 'Accuracy', 'F1 Score']
    palette = sns.color_palette("husl", len(datasets))
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        bar_width = 0.3  
        all_labels = sorted(set().union(*(data.keys() for data in datasets.values())))
        index = np.arange(len(all_labels))

        for i, (dataset, data) in enumerate(datasets.items()):
            values = [data.get(label, {}).get(metric, 0) for label in all_labels]
            bar_positions = index + i * bar_width
            ax.bar(bar_positions, values, bar_width, label=dataset, color=palette[i])

        ax.set_xlabel('Attributes', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=14, fontweight='bold')
        ax.set_xticks(index + (len(datasets) - 1) * bar_width / 2)
        ax.set_xticklabels(all_labels)

        ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(datasets))

        ax.grid(False)

        plt.savefig(f"{metric.lower()}_chart.png")


def plot_training_time():

    df = pd.read_csv("results/symbolic_results/training_time.csv")

    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 8))

    box_plot = sns.boxplot(x='Training_Task', y='Training_Time', data=df, width=0.3, 
                           boxprops=dict(facecolor='none', edgecolor='black'), 
                           medianprops=dict(color='blue'))


    plt.xlabel('Tasks', fontsize=14)
    plt.ylabel('Training Time (s)', fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.ylim(0, 1000) 
    plt.yticks(range(0, 1001, 50))  

    plt.savefig("box_plot.png")

if __name__ == '__main__':
    draw_ari_graph()