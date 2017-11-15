import torch
import torchvision
import argparse
import importlib
import os
import numpy as np

def main(args):
    #network = importlib.import_module(args.model_def)
    network = torchvision.models.squeezenet1_0()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    for name,m in network.named_modules():
        m.register_buffer('name', name)

        


def hook_model(network, hook):
    handler_list = []
    for m in network.models():
        handler = m.register_forward_hook(hook)
        handler_list.append(handler)
    return handler_list

def remove_hook(handler_list):
    for handler in handler_list:
        handler.remove()

def vis_hook(model, input, output):
    log_root = os.path.expanduser(args.vis_output_dir)
    path_token_list = model.name.split('.')
    log_dir = log_root
    for path_token in path_token_list:
        log_dir = os.path.join(log_dir, path_token)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'vis_data.npy')
    print('Writing visualization data into ', log_path)
    np.save(log_path, output.data.numpy())

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--vis_output_dir', type=str, help='TODO', default='~/visualization/output')
    parser.add_argument('--model_def', type=str, help='TODO', default='torchvision.models.SqueezeNet')
    parser.add_argument('--disable-gpu', dest='disable-gpu', default=False, action='store_true')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
