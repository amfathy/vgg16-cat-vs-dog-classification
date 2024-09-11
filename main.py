from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load the VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Build your model on top of the VGG16 base
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define paths to your train and test directories
train_dir = r'D:\College matrial\MultiMedia Minning\assigNMENTS\VGG16\CATvsDOG\train'
test_dir = r'D:\College matrial\MultiMedia Minning\assigNMENTS\VGG16\CATvsDOG\test1'

# Initialize ImageDataGenerator for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Initialize ImageDataGenerator for test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Generate batches of training data from the directories, taking 1000 samples (500 from each class)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Use only a portion of the dataset for training
)

# Generate batches of test data from the directories, taking 500 samples (250 from each class)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use only a portion of the dataset for testing
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Evaluate the model on the test data
evaluation = model.evaluate(test_generator)

# Print accuracy
print("Test Accuracy:", evaluation[1])