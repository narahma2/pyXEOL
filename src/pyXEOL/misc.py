import os


def create_folder(new_folder):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    return new_folder
