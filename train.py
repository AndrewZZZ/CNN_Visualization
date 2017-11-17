import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision
import argparse
import importlib
import os, sys
import numpy as np
import datetime
from ms_face_dataset import MSFaceDataset
import gc
import resource

def main(args):
    #network = importlib.import_module(args.model_def)
    train_dataset = MSFaceDataset(args.data_dir)
    vis_dataset = MSFaceDataset(args.vis_data_dir)
    #network = torchvision.models.squeezenet1_1(pretrained=True, num_classes = train_dataset.num_classes)
    network = torchvision.models.squeezenet1_1(pretrained=True, num_classes = 1000)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    counter = 0
    for name,m in network.named_modules():
        m.register_buffer('name', torch.IntTensor([counter]))
        counter += 1

    vis_test = False
    if vis_test == True:    
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=2)
        test_iter = iter(test_loader)
        test_batch = test_iter.next()
        test_hook_handler_list = hook_model(network, vis_hook)
        test_randn = torch.randn(10,3,255,255)
        print(type(test_randn))
        #network(Variable(test_batch['image']))
        network(Variable(test_batch['image']).float())
        #network(Variable(test_randn))

    vis_loader = torch.utils.data.DataLoader(vis_dataset, batch_size=10, shuffle=False, num_workers=1)
    vis_iter = iter(vis_loader)
    vis_batch = vis_iter.next()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    network.cuda()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(500):
        running_loss = 0.0
        scheduler.step()
        for i_batch, batch in enumerate(train_loader):
            images, labels = Variable(batch['image'].cuda()), Variable(batch['label'].cuda())
            optimizer.zero_grad()
            outputs = network(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if i_batch % 20 == 20 - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 20 ))
                running_loss = 0.0
            #if i_batch % 50 == 50-1:
        if epoch % 1 == 0:
            get_resource()
            print('Visualizing....')
            network.cpu()
            hook_handler_list = hook_model(network, vis_hook)
            network(Variable(vis_batch['image']))
            for hdlr in hook_handler_list:
                hdlr.remove()
            network.cuda()
            get_resource()



def get_resource():
    
    usage = resource.getrusage(resource.RUSAGE_SELF)
    for name, desc in [
    ('ru_utime', 'User time'),
    ('ru_stime', 'System time'),
    ('ru_maxrss', 'Max. Resident Set Size'),
    ('ru_ixrss', 'Shared Memory Size'),
    ('ru_idrss', 'Unshared Memory Size'),
    ('ru_isrss', 'Stack Size'),
    ('ru_inblock', 'Block inputs'),
    ('ru_oublock', 'Block outputs'),
    ]:
        print('%-25s (%-10s) = %s' % (desc, name, getattr(usage, name)))
                


def hook_model(network, hook):
    handler_list = []
    for m in network.modules():
        handler = m.register_forward_hook(hook)
        handler_list.append(handler)
    return handler_list

def remove_hook(handler_list):
    for handler in handler_list:
        handler.remove()

def vis_hook(model, input, output):
    # log_root = os.path.expanduser(args.vis_output_dir)
    # path_token_list = model.name.split('.')
    # log_dir = log_root
    # for path_token in path_token_list:
    #     log_dir = os.path.join(log_dir, path_token)
    # if not os.path.isdir(log_dir):
    #     os.makedirs(log_dir)
    #log_path = os.path.join(log_dir, 'vis_data.npy')
    #log_root = os.path.expanduser(args.vis_output_dir)
    log_root = os.path.expanduser('~/visualization/output')
    if not os.path.isdir(log_root):
         os.makedirs(log_root)
    #path_token_list = model.name.split('.')
    log_dir = log_root
    current_time = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    log_output_path = os.path.join(log_dir, str(model.name[0]) + '.' + current_time + '.npy')
    #log_input_path = os.path.join(log_dir, str(model.name[0]) + 'input' +'.' + current_time + '.npy')
    #print('Writing visualization data into ', log_output_path)
    np.save(log_output_path, mat2byte(output.data.numpy()))
    print(output.data.numpy().max())
    # print('Writing visualization data into ', log_input_path)
    # np.save(log_input_path, mat2byte(input[0].data.numpy()))
    

def mat2byte(mat):
    byte_mat = mat.copy()
    byte_mat -= byte_mat.min()
    byte_mat /= byte_mat.max()
    byte_mat *= 255
    return byte_mat.astype('ubyte')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='TODO', default='~/datasets/ms_face_data/1000_face_aligned')
    parser.add_argument('--vis_data_dir', type=str, help='TODO', default='~/datasets/ms_face_data/1000_face_aligned_val')
    parser.add_argument('--vis_output_dir', type=str, help='TODO', default='~/visualization/output')
    parser.add_argument('--model_def', type=str, help='TODO', default='torchvision.models.SqueezeNet')
    parser.add_argument('--disable-gpu', dest='disable-gpu', default=False, action='store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
