import numpy as np
import os
import matplotlib.pyplot as plt
import argparse, sys

def main(args):
    layer_name = args.layer_name
    image_num = args.image_num
    feature_map_num = args.feature_map_num
    load_list = []
    file_list = os.listdir('/home/andrew/visualization/output/')
    for file_name in file_list:
        if file_name.split('.')[0] == str(layer_name):
            load_list.append(file_name.split('.')[1])
    load_list = np.array(load_list).astype('int')
    load_list.sort()
    load_list = load_list.astype('str')

    image_list = []

    for time in load_list:
        data = np.load(os.path.join('/home/andrew/visualization/output/', str(layer_name) + '.' + time + '.npy'))
        image_list.append(data[image_num, feature_map_num, :, :])

    plt.ion()
    im = plt.imshow(image_list[0], cmap='gray', animated=True)
    for image in image_list:
        plt.pause(0.1)
        im.set_data(image)
        plt.draw()
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('layer_name', type=str, default='2')
    parser.add_argument('image_num', type=int, default=0)
    parser.add_argument('feature_map_num', type=int, default=0)
    return parser.parse_args(argv)
        
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
