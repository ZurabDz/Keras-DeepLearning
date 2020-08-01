import os
import random
import shutil


def _mkdir_train_test_valid(src: str, dest: str, name: str='processed'):
  dir_name = src.split('/')[-1]
  train_dir = os.path.join(dest, name, 'train', dir_name)
  test_dir = os.path.join(dest, name, 'test', dir_name)
  valid_dir = os.path.join(dest, name, 'valid', dir_name)
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(test_dir, exist_ok=True)
  os.makedirs(valid_dir, exist_ok=True)

  return train_dir, test_dir, valid_dir

def _shuffle_images(src: str):
  images = [os.path.join(src, photo) for photo in os.listdir(src)]
  random.shuffle(images)

  return images

def _split_images(images: list):
  size = len(images)

  train = images[:int(size * 0.6)]
  test = images[int(size * 0.6) : int(size * 0.9)]
  valid = images[int(size * 0.9) : ]

  return train, test, valid

def _copy_photoes_from_to(src: str, dest: str, photoes: list):
  for photo in photoes:
    shutil.copy2(photo, dest)

def train_test_valid_split(src: str, dest: str):
  train_dir, test_dir, valid_dir = _mkdir_train_test_valid(src, dest)
  images = _shuffle_images(src)
  train, test, valid = _split_images(images)
  _copy_photoes_from_to(src, train_dir, train)
  _copy_photoes_from_to(src, test_dir, test)
  _copy_photoes_from_to(src, valid_dir, valid)


train_test_valid_split('data/RickAndMorty/meeseeks', 'data')
train_test_valid_split('data/RickAndMorty/morty', 'data')
train_test_valid_split('data/RickAndMorty/poopybutthole', 'data')
train_test_valid_split('data/RickAndMorty/rick', 'data')
train_test_valid_split('data/RickAndMorty/summer', 'data')