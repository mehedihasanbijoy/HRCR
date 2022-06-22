import glob
import os
import pandas as pd


def train_test_dfs():
    train_images = glob.glob('./Dataset/Train/*/*.*')
    train_paths = [img.replace('\\', '/') for img in train_images]
    train_cls = [int(img.split('/')[3]) for img in train_paths]
    test_images = glob.glob('./Dataset/Test/*/*.*')
    test_paths = [img.replace('\\', '/') for img in test_images]
    test_cls = [int(img.split('/')[3]) for img in test_paths]
    train_df = pd.DataFrame({
        'Path': train_paths, 'Cls': train_cls
    })
    test_df = pd.DataFrame({
        'Path': test_paths, 'Cls': test_cls
    })
    train_df.to_csv('./dfs/train.csv', index=False)
    test_df.to_csv('./dfs/test.csv', index=False)
    return train_df, test_df


def check_cmaterdb():
    image_paths = glob.glob('./Dataset/*/*/*.*')
    paths = []
    # print(len(image_paths), image_paths[0])
    for path in image_paths:
        ext = path.split('.')[-1]
        if ext != 'bmp':
            # print(path)
            os.remove(path)


if __name__ == '__main__':
    check_cmaterdb()
    train_df, test_df = train_test_dfs()
    print(len(train_df), len(test_df))
    # print(train_df)