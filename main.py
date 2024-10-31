# main.py
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_load import load_data
from model import create_model
from test import evaluate_model

def main():
    # Data Load
    ns1, ns2, ns3 = 5000 * 8, 500, 500
    train_images, train_labels, test_images, test_labels, valid_images, valid_labels = load_data(ns1, ns2, ns3)

    # Reshape images
    train_images = train_images.reshape((2 * ns1, 80, 80, 1)).astype('float32')
    test_images = test_images.reshape((2 * ns2, 80, 80, 1)).astype('float32')
    valid_images = valid_images.reshape((2 * ns3, 80, 80, 1)).astype('float32')
    
    # Model creation
    model = create_model(input_shape=(80, 80, 1))
    model.summary()

    # Define Callbacks
    checkpoint = ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    # Model training
    history = model.fit(
        train_images, train_labels, epochs=30, batch_size=128, 
        validation_data=(valid_images, valid_labels), callbacks=[checkpoint, early_stopping]
    )

    # Save training metrics
    with open("history_metrics.dat", 'w') as f:
        for epoch, metrics in enumerate(zip(history.history['binary_accuracy'], history.history['auc'], 
                                            history.history['true_positives'], history.history['true_negatives'],
                                            history.history['false_positives'], history.history['false_negatives'], 
                                            history.history['val_binary_accuracy'], history.history['val_auc'], 
                                            history.history['val_true_positives'], history.history['val_true_negatives'],
                                            history.history['val_false_positives'], history.history['val_false_negatives'])):
            f.write(f"{epoch + 1} " + " ".join(map(str, metrics)) + "\n")

    with open("history_losses.dat", 'w') as f:
        for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            f.write(f"{epoch + 1} {loss} {val_loss}\n")

    # Model evaluation
    evaluate_model(model, test_images, test_labels)

if __name__ == "__main__":
    main()

