import torch
from torch.utils.data import TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt

# --- EVALUATION ----

def _accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def evaluate(model, test_dataloader):
    model.eval()

    loss_arr = []
    accuracy_arr = []

    device = "cuda"

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2]}

            outputs = model(**inputs)
            loss, logits = outputs[:2]

            preds = torch.argmax(logits, dim=1)
            labels = inputs['labels']

            loss_arr.append(loss.item())
            accuracy_arr.append(_accuracy(preds, labels))

    model.train()
    avg_loss = sum(loss_arr) / len(loss_arr)
    avg_accuracy = sum(accuracy_arr) / len(accuracy_arr)
    return avg_loss, avg_accuracy

# --- PARTITIONING ---

def partition_dataset_homogeneous(dataset: TensorDataset, num_clients=3):
    indices = np.random.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, split) for split in splits]

def partition_dataset_heterogeneous(dataset: TensorDataset):
    """ 3-class classification task, partitioning scheme:
      - Client 0: 90% of samples have label 0, 5% label 1, 5% label 2
      - Client 1: 90% label 1, 5% label 0, 5% label 2
      - Client 2: 90% label 2, 5% label 0, 5% label 1
    """
    # Extract labels as numpy array
    labels = np.array([sample.item() for sample in dataset.tensors[2]])
    indices = np.arange(len(labels))
    partitions = []
    
    proportions = [
        {0: 0.9, 1: 0.05, 2: 0.05},
        {1: 0.9, 0: 0.05, 2: 0.05},
        {2: 0.9, 0: 0.05, 1: 0.05},
    ]
    for client in range(len(proportions)):
        client_indices = []
        for label in [0, 1, 2]:
            label_indices = indices[labels == label]
            # Determine number of samples to pick for this label from client partition
            num_samples = int(len(label_indices) * proportions[client][label])
            if num_samples > 0:
                selected = np.random.choice(label_indices, size=num_samples, replace=False)
                client_indices.extend(selected)
        partitions.append(Subset(dataset, client_indices))
    return partitions

# --- PLOTS ---

def plot_loss_and_accuracy(
        loss_history_lora,
        loss_history_ffa_lora,
        acc_history_lora,
        acc_history_ffa_lora,
):

    rounds = range(1, len(loss_history_lora)+1)

    plt.figure(figsize=(14,6))

    # Plot training loss
    plt.subplot(1,2,1)
    plt.plot(rounds, loss_history_lora, label='LoRA', color='blue', marker='o', markersize=3)
    plt.plot(rounds, loss_history_ffa_lora, label='FFA-LoRA', color='red', marker='s', markersize=3)
    plt.xlabel("Global Round")
    plt.ylabel("Loss")
    plt.title("Evaluation Loss Over Rounds")
    plt.legend()

    # Plot evaluation accuracy
    plt.subplot(1,2,2)
    plt.plot(rounds, acc_history_lora, label='LoRA', color='blue', marker='o', markersize=3)
    plt.plot(rounds, acc_history_ffa_lora, label='FFA-LoRA', color='red', marker='s', markersize=3)
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy Over Rounds")
    plt.legend()

    plt.tight_layout()
    plt.show()