import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_loader import load_mnist_data
from models import build_mlp_model

MNIST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reduced MNIST Data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_prob1_experiments():
    print("==========================================================")
    print("   STARTING PROBLEM 1: AUTOMATED EXPERIMENT PIPELINE")
    print("==========================================================\n")

    x_train, y_train, x_test, y_test = load_mnist_data(MNIST_DATA_DIR)
    input_size = x_train.shape[1] * x_train.shape[2]
    
    x_train_flat = x_train.reshape((-1, input_size))
    x_test_flat = x_test.reshape((-1, input_size))
    
    # --------------------------------------------------------------------------
    # DEFINE THE CONFIGURATIONS REQUIRED BY THE ASSIGNMENT
    # --------------------------------------------------------------------------
    experiments = [
        {"name": "Exp1_1_Hidden_Layer", "layers": 1, "reg": True, "lr": 0.001},
        {"name": "Exp2_3_Hidden_Layers", "layers": 3, "reg": True, "lr": 0.001},
        {"name": "Exp3_4_Hidden_Layers", "layers": 4, "reg": True, "lr": 0.001},
        {"name": "Exp4_No_Regularization", "layers": 3, "reg": False, "lr": 0.001},
        {"name": "Exp5_High_Learning_Rate", "layers": 3, "reg": True, "lr": 0.01}
    ]

    report_path = os.path.join(RESULTS_DIR, 'prob1_comparative_report.txt')
    
    # Open report file to log results dynamically
    with open(report_path, 'w', encoding='utf-8') as report:
        report.write("====================================================\n")
        report.write("    PROBLEM 1: MLP COMPARATIVE ANALYSIS REPORT      \n")
        report.write("====================================================\n")
        report.write(f"Student Name: بيريهان سلطان\n\n")

        # Iterate through each experimental setup
        for config in experiments:
            exp_name = config["name"]
            print(f"\n---> RUNNING: {exp_name} <---")
            
            # Build the model dynamically based on current dictionary
            model = build_mlp_model(
                input_dim=input_size, 
                num_layers=config["layers"], 
                use_regularization=config["reg"], 
                learning_rate=config["lr"]
            )
            
            early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0)
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)

            import time
            start_train = time.time()
            history = model.fit(
                x_train_flat, y_train,
                epochs=30, batch_size=128,
                validation_data=(x_test_flat, y_test),
                callbacks=[early_stop, lr_scheduler],
                verbose=0 # Set to 0 to keep the terminal clean during multiple runs
            )
            val_train_time = (time.time() - start_train) * 1000

            start_test = time.time()
            test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
            val_test_time = (time.time() - start_test) * 1000
            
            print(f"Finished {exp_name} -> Acc: {test_acc*100:.2f}% | Train Time: {val_train_time:.1f}ms | Test Time: {val_test_time:.1f}ms")
            
            # --- Save Individual Plots ---
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Acc')
            plt.plot(history.history['val_accuracy'], label='Val Acc', linestyle='--')
            plt.title(f'{exp_name} - Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--')
            plt.title(f'{exp_name} - Loss')
            plt.legend()
            
            plt.savefig(os.path.join(RESULTS_DIR, f'{exp_name}_Curves.png'))
            plt.close()

            # --- Log to Report ---
            report.write(f"Configuration: {exp_name}\n")
            report.write(f"- Hidden Layers: {config['layers']}\n")
            report.write(f"- Regularization (Drop+BN): {'Yes' if config['reg'] else 'No'}\n")
            report.write(f"- Learning Rate: {config['lr']}\n")
            report.write(f"- Final Test Accuracy: {test_acc * 100:.2f}%\n")
            report.write(f"- Training Time (ms): {val_train_time:.1f}\n")
            report.write(f"- Testing Time (ms): {val_test_time:.1f}\n")
            report.write(f"- Final Test Loss: {test_loss:.4f}\n")
            report.write("-" * 52 + "\n")
            
            print(f"Finished {exp_name} -> Accuracy: {test_acc * 100:.2f}%")

    print(f"\n[SUCCESS] All experiments done! Plots and comparative report saved in '{RESULTS_DIR}'.")

if __name__ == "__main__":
    run_prob1_experiments()