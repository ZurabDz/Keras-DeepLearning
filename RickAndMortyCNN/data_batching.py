from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = 'data/processed/train'
test_dir = 'data/processed/test'
valid_dir = 'data/processed/valid'

train_datagen = ImageDataGenerator(
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  fill_mode='nearest',
  horizontal_flip=True,
  vertical_flip=True,
  rescale=1./255
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size=(256, 256),
  batch_size=32,
  class_mode='categorical'
)

valid_generator = test_datagen.flow_from_directory(
  valid_dir,
  target_size=(256, 256),
  batch_size=32,
  class_mode='categorical'
)