import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import SpeechDataset
from model import SimpleSpeechCNN

# Helper function to train the given model
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    start_time = time.time()
    
    history_loss = []
    history_acc = []
    
    for epoch in range(num_epochs):
        curr_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 1. Forward Pass: compute outputs
            outputs = model(inputs)
            # 2. Compute Loss
            loss = criterion(outputs, labels)
            
            # 3. Backward pass & Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = curr_loss / total
        epoch_acc = 100 * correct / total
        
        history_loss.append(epoch_loss)
        history_acc.append(epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.1f}%")
        
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total Training Time: {training_time:.2f} seconds")
    return training_time, history_loss, history_acc

# Helper function to evaluate the given model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward Pass: compute predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    end_time = time.time()
    testing_time = end_time - start_time
    accuracy = 100 * correct / total
    
    print(f"Test Accuracy: {accuracy:.1f}%")
    print(f"Total Testing Time: {testing_time:.2f} seconds")
    return accuracy, testing_time

def run_experiment(experience_name, data_dir, audio_aug, image_aug):
    print(f"\n{'='*50}\nStarting Experiment: {experience_name}")
    print(f"Audio Augmentation: {audio_aug} | Image Augmentation: {image_aug}")
    
    # Use GPU hardware acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Define hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 15

    # 1. Load Data
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")
    
    train_dataset = SpeechDataset(data_dir=train_dir, audio_aug=audio_aug, image_aug=image_aug)
    test_dataset = SpeechDataset(data_dir=test_dir, audio_aug=False, image_aug=False) # We never augment the test set!
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Define Model, Loss, Optimizer
    model = SimpleSpeechCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Train
    print("Training phase...")
    train_time, hw_loss, hw_acc = train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
    
    # 4. Evaluate
    print("\nTesting phase...")
    test_acc, test_time = evaluate_model(model, test_loader, device)

    # 5. Save Learning Curve figure
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(hw_loss, label='Loss')
    plt.title(f"{experience_name}: Loss")
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(hw_acc, label='Accuracy', color='orange')
    plt.title(f"{experience_name}: Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    os.makedirs("results", exist_ok=True)
    safe_name = experience_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"results/{safe_name}_curve.png")
    plt.close()

    return test_acc, train_time, test_time


if __name__ == "__main__":
    # Ensure this absolute path points exactly to where the audio-dataset lives
    AUDIO_DATA_DIR = r"D:\EECE4\NeuralNetworks\repo-clone\Materials\audio-dataset"
    
    # Create results directory to save figures and final summary text file
    os.makedirs("results", exist_ok=True)
    
    # Part 3a: Baseline (no augmentations)
    acc_a, train_a, test_a = run_experiment(
        experience_name="Baseline (Part A)", data_dir=AUDIO_DATA_DIR, audio_aug=False, image_aug=False)

    # Part 3b: Audio Augmentation
    acc_b, train_b, test_b = run_experiment(
        experience_name="Audio Augmentation (Part B)", data_dir=AUDIO_DATA_DIR, audio_aug=True, image_aug=False)

    # Part 3c: Image Augmentation (Spectrogram squeezing/expanding)
    acc_c, train_c, test_c = run_experiment(
        experience_name="Image Augmentation (Part C)", data_dir=AUDIO_DATA_DIR, audio_aug=False, image_aug=True)

    # Part 3d: Combined Augmentations
    acc_d, train_d, test_d = run_experiment(
        experience_name="Combined Augmentations (Part D)", data_dir=AUDIO_DATA_DIR, audio_aug=True, image_aug=True)

    summary_text = (
        "================== FINAL SUMMARY ==================\n"
        f"A) Baseline Acc: {acc_a:.1f}% | Train Time: {train_a:.2f}s | Test Time: {test_a:.2f}s\n"
        f"B) Audio Aug Acc: {acc_b:.1f}% | Train Time: {train_b:.2f}s | Test Time: {test_b:.2f}s\n"
        f"C) Image Aug Acc: {acc_c:.1f}% | Train Time: {train_c:.2f}s | Test Time: {test_c:.2f}s\n"
        f"D) Combined Acc : {acc_d:.1f}% | Train Time: {train_d:.2f}s | Test Time: {test_d:.2f}s\n"
    )

    print(summary_text)

    # Save summary logic explicitly mapped to file
    with open("results/final_results.txt", "w") as f:
        f.write(summary_text)

    print("\n[INFO] All visual curves and the text summary have been saved inside the 'results' folder!")
