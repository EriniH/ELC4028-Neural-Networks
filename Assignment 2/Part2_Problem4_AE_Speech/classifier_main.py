import os
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AE_RESULTS_DIR = os.path.join(BASE_DIR, "results")
CLASSIFIER_RESULTS_DIR = os.path.join(BASE_DIR, "results_after_classification")
os.makedirs(CLASSIFIER_RESULTS_DIR, exist_ok=True)

def load_bottleneck_features(source_dir):
    x_train = np.load(os.path.join(source_dir, 'Bottleneck_Features_Train.npy'))
    x_test  = np.load(os.path.join(source_dir, 'Bottleneck_Features_Test.npy'))
    y_train = np.load(os.path.join(source_dir, 'Y_train_ae.npy'))
    y_test  = np.load(os.path.join(source_dir, 'Y_test_ae.npy'))
    return x_train, y_train, x_test, y_test

def build_dynamic_audio_classifier(input_dim, num_classes=10, num_layers=3, use_regularization=True, learning_rate=0.0005):
    """ Builds dynamic MLP for audio latent features comparison. """
    model = models.Sequential(name=f"Audio_MLP_{num_layers}_Layers")
    neurons = [512, 256, 128, 64]
    
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    for i in range(num_layers):
        model.add(layers.Dense(units=neurons[i], name=f"Hidden_{i+1}"))
        if use_regularization:
            model.add(layers.BatchNormalization(name=f"BatchNorm_{i+1}"))
        model.add(layers.Activation('relu', name=f"Relu_{i+1}"))
        if use_regularization:
            model.add(layers.Dropout(rate=0.2, name=f"Dropout_{i+1}"))
            
    model.add(layers.Dense(units=num_classes, activation='softmax', name="Output_Layer"))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_audio_experiments():
    print("==========================================================")
    print("   STARTING PROBLEM 4: AUTOMATED EXPERIMENT PIPELINE")
    print("==========================================================\n")

    x_train, y_train, x_test, y_test = load_bottleneck_features(AE_RESULTS_DIR)
    feature_size = x_train.shape[1] 

    # --------------------------------------------------------------------------
    # LIST OF CONFIGURATIONS TO SATISFY REQUIREMENTS
    # --------------------------------------------------------------------------
    experiments = [
        {"name": "Audio_Exp1_1_Layer", "layers": 1, "reg": True, "lr": 0.0005},
        {"name": "Audio_Exp2_3_Layers", "layers": 3, "reg": True, "lr": 0.0005},
        {"name": "Audio_Exp3_4_Layers", "layers": 4, "reg": True, "lr": 0.0005},
        {"name": "Audio_Exp4_No_Reg", "layers": 3, "reg": False, "lr": 0.0005},
        {"name": "Audio_Exp5_High_LR", "layers": 3, "reg": True, "lr": 0.01}
    ]

    report_path = os.path.join(CLASSIFIER_RESULTS_DIR, 'prob4_comparative_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as report:
        report.write("====================================================\n")
        report.write("    PROBLEM 4: AUDIO MLP COMPARATIVE REPORT         \n")
        report.write("====================================================\n")
        report.write(f"Student Name: بيريهان سلطان\n\n")

        for config in experiments:
            exp_name = config["name"]
            print(f"\n---> RUNNING: {exp_name} <---")
            
            model = build_dynamic_audio_classifier(
                input_dim=feature_size, 
                num_layers=config["layers"], 
                use_regularization=config["reg"], 
                learning_rate=config["lr"]
            )
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, verbose=0)

            history = model.fit(
                x_train, y_train,
                epochs=40, batch_size=64,
                validation_data=(x_test, y_test),
                callbacks=[early_stop, lr_scheduler],
                verbose=0
            )

            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            
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
            
            plt.savefig(os.path.join(CLASSIFIER_RESULTS_DIR, f'{exp_name}_Curves.png'))
            plt.close()

            report.write(f"Configuration: {exp_name}\n")
            report.write(f"- Hidden Layers: {config['layers']}\n")
            report.write(f"- Regularization: {'Yes' if config['reg'] else 'No'}\n")
            report.write(f"- Learning Rate: {config['lr']}\n")
            report.write(f"- Final Test Accuracy: {test_acc * 100:.2f}%\n")
            report.write("-" * 52 + "\n")
            
            print(f"Finished {exp_name} -> Accuracy: {test_acc * 100:.2f}%")

    print(f"\n[SUCCESS] Audio comparisons completed! Check '{CLASSIFIER_RESULTS_DIR}'.")

if __name__ == "__main__":
    run_audio_experiments()