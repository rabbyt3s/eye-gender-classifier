import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

def create_data_flow():
    try:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            horizontal_flip=True
        )

        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training',
            classes=['male', 'female']
        )

        val_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            classes=['male', 'female']
        )
        
        print("\n✅ Data flow created successfully")
        print(f"Classes: {train_generator.class_indices}")
        return train_generator, val_generator
        
    except Exception as e:
        print(f"\n❌ Error creating data flow: {str(e)}")
        return None, None

def create_simple_model(input_shape):
    try:
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n✅ Model created successfully")
        model.summary()
        return model
        
    except Exception as e:
        print(f"\n❌ Error creating model: {str(e)}")
        return None

def train_model(model, train_gen, val_gen):
    try:
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=3),
            callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),  
            callbacks.CSVLogger('training_log.csv')
        ]
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("\n✅ Training completed successfully")
        return history
        
    except Exception as e:
        print(f"\n❌ Training error: {str(e)}")
        return None

if __name__ == "__main__":
    train_gen, val_gen = create_data_flow()
    
    if train_gen and val_gen:
        model = create_simple_model((IMG_SIZE, IMG_SIZE, 3))
        
        if model:
            history = train_model(model, train_gen, val_gen)
            
            if history:
                print("\n📊 Final metrics:")
                print(f"Training accuracy: {history.history['accuracy'][-1]:.2f}")
                print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.2f}")
                print(f"Training loss: {history.history['loss'][-1]:.4f}")
                print(f"Validation loss: {history.history['val_loss'][-1]:.4f}")
