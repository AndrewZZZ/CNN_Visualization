import torch
import torchvision
import argparse
import importlib


def main(args):
    #network = importlib.import_module(args.model_def)
    network = torchvision.models.SqueezeNet
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    

def vis_hook(model, input, output):
    
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--vis_output_dir', type=str, help='TODO', default='~/visualization/output')
    parser.add_argument('--model_def', type=str, help='TODO', default='torchvision.models.SqueezeNet')
    parser.add_argument('--disable-gpu', dest='disable-gpu', default=False, action='store_true')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
