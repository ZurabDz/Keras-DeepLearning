import os
import random
import shutil


def _mkdir_train_test_valid(src: str, dest: str, name: str = 'processed'):
    '''
    src: Takes Directory, which will be splitted into 3 parts.
    dest: Path where train, test, valid dirs will be created
    e.g src -> /home/username/data/flower
        dest-> /home/username/data

    it will create /home/username/data/processed/{x}
    x will be train, test and valid respectively
    '''
    dir_name = src.split('/')[-1]
    train_dir = os.path.join(dest, name, 'train', dir_name)
    test_dir = os.path.join(dest, name, 'test', dir_name)
    valid_dir = os.path.join(dest, name, 'valid', dir_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    return train_dir, test_dir, valid_dir


def _get_images(src: str, shuffle: bool):
    """
    src: Takes source of directory
    returns: pathes of all images as a python list
    shuffle: boolean type for shuffling or not
    """
    images = [os.path.join(src, photo) for photo in os.listdir(src)]
    if shuffle:
        random.shuffle(images)

    return images


def _split_images(images: list):
    """
    Splits list of pathes into training, test and valid sets.

    images: list of photo pathes.(use _get_images first for list)
    return: Returns tuple of 3 python lists
    """

    # TODO: Split ratio is not customizable
    size = len(images)
    train = images[:int(size * 0.6)]
    test = images[int(size * 0.6): int(size * 0.9)]
    valid = images[int(size * 0.9):]

    return train, test, valid


def _copy_photoes_from_to(dest: str, photoes: list):
    """
    In house function for copying files to destination

    dest: path of destination directory
    photoes: photoes pathes' list
    """
    for photo in photoes:
        shutil.copy2(photo, dest)


def train_test_valid_split(src: str, dest: str, shuffle: bool = True):
    '''
    Naive approach to split given directory's photoes/files into
    train, test and valid directories. Ratio is not customizable yet.

    src: path of directory, which needs to be splitted
    dest: destination where train, test, valid directories will be created
    e.g destination/preprocessed/{train, test, valid}
    '''
    train_dir, test_dir, valid_dir = _mkdir_train_test_valid(src, dest)
    images = _get_images(src, shuffle)
    train, test, valid = _split_images(images)
    _copy_photoes_from_to(train_dir, train)
    _copy_photoes_from_to(test_dir, test)
    _copy_photoes_from_to(valid_dir, valid)


if __name__ == "__main__":
    train_test_valid_split('data/RickAndMorty/meeseeks', 'data')
    train_test_valid_split('data/RickAndMorty/morty', 'data')
    train_test_valid_split('data/RickAndMorty/poopybutthole', 'data')
    train_test_valid_split('data/RickAndMorty/rick', 'data')
    train_test_valid_split('data/RickAndMorty/summer', 'data')
