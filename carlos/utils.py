from fileinput import filename
import os
from tarfile import TarInfo
from tkinter.constants import S
from botbowl.core.model import Action
import torch
import numpy as np
# from agents.carlos_agent import CNNPolicy
from botbowl.ai.env import BotBowlEnv
import gym
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import carlos

def get_data_path(rel_path):
    root_dir = carlos.__file__.replace("__init__.py", "")
    filename = os.path.join(root_dir, "data/" + rel_path)
    return os.path.abspath(os.path.realpath(filename))
 
def create_dataset():
    pairs_directory = get_data_path('pairs')
    if not os.path.exists(pairs_directory):
        print(f"{pairs_directory} doesn't exist")
        return
 
    dataset = {
        'X_spatial': [],
        'X_non_spatial': [],
        'Y': []
    }
 
    for f, file in enumerate(os.listdir(pairs_directory)):
        if f > 400000:
            break
        filename = os.path.join(pairs_directory, file)
 
        pair = torch.load(filename)
        print('Loadded ', (filename))
        dataset['X_spatial'].append(pair['obs']['spatial_obs'])
        dataset['X_non_spatial'].append(pair['obs']['non_spatial_obs'])
        dataset['Y'].append(pair['actions'])
 
 
    # print(dataset)
    dataset_directory = get_data_path("dataset")
    if not os.path.exists(dataset_directory):
        os.mkdir(dataset_directory)
    filename = os.path.join(dataset_directory, f"dataset.pt")
    torch.save(dataset, filename)


def get_action_type(action_idx, non_spatial_action_types, spatial_action_types, board_squares):
    if action_idx < len(non_spatial_action_types):
        return non_spatial_action_types[action_idx]
    spatial_idx = action_idx - len(non_spatial_action_types)
    spatial_action_type_idx = int(spatial_idx / board_squares)
    return spatial_action_types[spatial_action_type_idx]


def actions_stats():
    directory = get_data_path('dataset')
    if not os.path.exists(directory):
        os.mkdir(directory)
 
    env = gym.make('FFAI-v3')
    spatial_obs_space = env.observation_space.spaces['board'].shape
    board_dim = (spatial_obs_space[1], spatial_obs_space[2])
    board_squares = spatial_obs_space[1] * spatial_obs_space[2]
 
    non_spatial_obs_space = env.observation_space.spaces['state'].shape[0] + env.observation_space.spaces['procedures'].shape[0] + env.observation_space.spaces['available-action-types'].shape[0]
    non_spatial_action_types = BotBowlEnv.simple_action_types + BotBowlEnv.defensive_formation_action_types + BotBowlEnv.offensive_formation_action_types
    num_non_spatial_action_types = len(non_spatial_action_types)
    spatial_action_types = BotBowlEnv.positional_action_types
    num_spatial_action_types = len(spatial_action_types)
    num_spatial_actions = num_spatial_action_types * spatial_obs_space[1] * spatial_obs_space[2]
    action_space = num_non_spatial_action_types + num_spatial_actions
 
    actions_dic = {}
    for action_type in non_spatial_action_types:
        actions_dic[action_type] = 0
    for action_type in spatial_action_types:
        actions_dic[action_type] = 0
    pairs_count = 0

    directory = get_data_path('dataset')
    if not os.path.exists(directory):
        return
   
    filename = os.path.join(directory, f"dataset.pt")
    dataset = torch.load(filename)

    for a, action in enumerate(dataset['Y']):
        if a % 200 == 0:
            print(f"{a}/{len(dataset['Y'])}")
        action_idx = action.type(torch.IntTensor).item()
    # for f, file in enumerate(os.listdir(directory)):
    #     if file == 'dataset.pt':
    #         continue
    #     filename = os.path.join(directory, file)
    #     print('Loadded ', (filename))
    #     pairs_count += 1
 
    #     pair = torch.load(filename)
    #     action_idx = pair['actions'][0].type(torch.IntTensor).item()
        # print(action_idx)
        action_type = get_action_type(action_idx, non_spatial_action_types, spatial_action_types, board_squares)
        # print(action_type)
        count =actions_dic.get(action_type)
        if count == None:
            count = 0
        actions_dic.update({action_type : count+1})
        # print(actions_dic.get(action_type))
 
    print(actions_dic)
    actions_percentages = {}
    for key in actions_dic.keys():
        type_count = actions_dic[key]
        actions_percentages[key] = (type_count / len(dataset['Y'])) * 100
   
    for key in actions_percentages.keys():
        print(f'{key}: {actions_percentages.get(key)}%')
    print(actions_percentages)

 
 
def load_example():
    directory = get_data_path('pairs')
    if not os.path.exists(directory):
        os.mkdir(directory)
 
    files = os.listdir(directory)
    file = files[1]
    # print(file)
 
    filename = os.path.join(directory, file)
    pair = torch.load(filename)
    # print(pair)
    return pair
 
 
def load_dataset():
    print('Loading dataset')
    directory = get_data_path('dataset')
    if not os.path.exists(directory):
        return
   
    filename = os.path.join(directory, f"dataset.pt")
    dataset = torch.load(filename)
    # dataset['X_spatial'] = dataset['X_spatial'][0:400]
    # dataset['X_non_spatial'] = dataset['X_non_spatial'][0:400]
    # # dataset['action_masks'] = dataset['action_masks'][0:400]
    # dataset['Y'] = dataset['Y'][0:400]
    print('Dataset loaded')
    return dataset
   
 
def make_trainset(dataset):
    print('Making trainset')
    # print(len(dataset['X_spatial']))
    spatial_obs = torch.stack(dataset['X_spatial'][0:split])
    # print(spatial_obs.size())
    spatial_obs = torch.reshape(spatial_obs, (split, 44, 17, 28)) # TODO: fix magic number
    non_spatial_obs = torch.stack(dataset['X_non_spatial'][0:split])
    non_spatial_obs = torch.reshape(non_spatial_obs, (split, 1, 116))
    # action_masks = torch.stack(dataset['action_masks'][0:split])
    # action_masks = torch.reshape(action_masks, (split, 8117))
    # print(action_masks.shape)
    actions = torch.stack(dataset['Y'][0:split])
    actions = torch.flatten(actions)
    actions = actions.long()
    # print(actions.size())
 
    trainset = torch.utils.data.TensorDataset(spatial_obs, non_spatial_obs, actions)
    print('Trainset made')
    return trainset
 
def make_testset(dataset):
    print('Makig testset')
    spatial_obs = torch.stack(dataset['X_spatial'][split:-1])
    spatial_obs = torch.reshape(spatial_obs, (len(spatial_obs), 44, 17, 28)) # TODO: Fix magic number
    non_spatial_obs = torch.stack(dataset['X_non_spatial'][split:-1])
    non_spatial_obs = torch.reshape(non_spatial_obs, (len(non_spatial_obs), 1, 116))
    # action_masks = torch.stack(dataset['action_masks'][split:-1])
    # action_masks = torch.reshape(action_masks, (len(action_masks), 8117))
    actions = torch.stack(dataset['Y'][split:-1])
    actions = torch.flatten(actions)
    actions = actions.long()
 
    testset = torch.utils.data.TensorDataset(spatial_obs, non_spatial_obs, actions)
    print('Testset made')
    return testset
 
 
def train(epoch, trainloader, model, device, optimizer, loss_function):
    print('\nEpoch : %d'%epoch)
    model.train()
    
    epoch_loss = []
    epoch_accu = []

    running_loss = 0
    correct = 0
    total = 0
 
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        spatial_obs, non_spatial_obs, actions = data
        spatial_obs = spatial_obs.to(device)
        non_spatial_obs = non_spatial_obs.to(device)
        actions = actions.to(device)
 
        # zero the parameter gradients
        optimizer.zero_grad()
 
        # forward + backward + optimize
        values, action_log_probs, = model.get_action_log_probs(spatial_obs, non_spatial_obs)
        # action_log_probs = torch.where(action_log_probs == float('-inf'), torch.tensor(0., device=device), action_log_probs)
        # print(spatial_obs.size(), non_spatial_obs.size(), actions.size(), outputs[1].size())
        # print(actions)
        loss = loss_function(action_log_probs, actions)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = action_log_probs.max(1)
        total += actions.size(0)
        correct += predicted.eq(actions).sum().item()

        epoch_loss.append(loss.item())
        epoch_accu.append(100* predicted.eq(actions).sum().item()/actions.size(0))
       
    train_loss=running_loss/len(trainloader)
    accu=100.*correct/total

    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))
 
    model_dir = get_data_path('models')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    PATH = os.path.join(model_dir, 'epoch-{}.pth'.format(epoch))
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': accu
                }, PATH)
    model.cuda()
    return train_loss, accu

 
def test(epoch, testloader, model, device, optimizer, loss_function):
  model.eval()
 
  epoch_loss = []
  epoch_accu = []

  running_loss=0
  correct=0
  total=0
 
  with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        spatial_obs, non_spatial_obs, actions = data
        spatial_obs = spatial_obs.to(device)
        non_spatial_obs = non_spatial_obs.to(device)
        actions = actions.to(device)
 
        # zero the parameter gradients
        optimizer.zero_grad()
 
        # forward + backward + optimize
        values, action_log_probs, = model.get_action_log_probs(spatial_obs, non_spatial_obs)
        # action_log_probs = torch.where(action_log_probs == float('-inf'), torch.tensor(0., device=device), action_log_probs)
        # print(spatial_obs.size(), non_spatial_obs.size(), actions.size(), outputs[1].size())
        # print(actions)
        loss = loss_function(action_log_probs, actions)
 
        running_loss += loss.item()
       
        _, predicted = action_log_probs.max(1)
        total += actions.size(0)
        correct += predicted.eq(actions).sum().item()
       
        epoch_loss.append(loss.item())
        epoch_accu.append(100* predicted.eq(actions).sum().item()/actions.size(0))
   
  test_loss=running_loss/len(testloader)
  accu=100.*correct/total
 
  print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
  return test_loss, accu

 
 
split = 300000
batch_size = 4
 
# Architecture
num_hidden_nodes = 1024
num_cnn_kernels = [128, 64, 17]

def make_env():
    env = BotBowlEnv()
    return env

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Device ', device)
 
    # create_dataset()
    dataset = load_dataset()
 
    trainset = make_trainset(dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
 
    testset = make_testset(dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
   
 
    env = make_env()
    env.reset()
    spat_obs, non_spat_obs, action_mask = env.get_state()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)

    # spatial_obs_space = env.observation_space.spaces['board'].shape
    # board_dim = (spatial_obs_space[1], spatial_obs_space[2])
    # board_squares = spatial_obs_space[1] * spatial_obs_space[2]
 
    # non_spatial_obs_space = env.observation_space.spaces['state'].shape[0] + env.observation_space.spaces['procedures'].shape[0] + env.observation_space.spaces['available-action-types'].shape[0]
    # non_spatial_action_types = NewBotBowlEnv.simple_action_types + NewBotBowlEnv.defensive_formation_action_types + NewBotBowlEnv.offensive_formation_action_types
    # num_non_spatial_action_types = len(non_spatial_action_types)
    # spatial_action_types = NewBotBowlEnv.positional_action_types
    # num_spatial_action_types = len(spatial_action_types)
    # num_spatial_actions = num_spatial_action_types * spatial_obs_space[1] * spatial_obs_space[2]
    # action_space = num_non_spatial_action_types + num_spatial_actions

    # filename = "epoch-19.pth"
    # state_dict_file = torch.load(filename)
    model = CNNPolicy(spatial_obs_space, non_spatial_obs_space, hidden_nodes=num_hidden_nodes, kernels=num_cnn_kernels, actions=action_space)
    # model.load_state_dict(state_dict_file['model_state_dict'])
   
    # pair = load_example()
    # spatial_obs = pair['obs']['spatial_obs']
    # non_spatial_obs = pair['obs']['non_spatial_obs']
    # action_idx = pair['actions']
 
    # print(spatial_obs)
    # print(non_spatial_obs)
    # print(action_idx)
 
    # values, actions = model.act(spatial_obs, non_spatial_obs, None)
    # print(values)
    # print(actions)
 
    model.to(device)
 
    loss_function = nn.NLLLoss()
    optimizer = optim.RAdam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    # optimizer.load_state_dict(state_dict_file['optimizer_state_dict'])
 
    train_losses=[]
    train_accu=[]
 
    eval_losses=[]
    eval_accu=[]

   
    for epoch in range(30):
        start_time = time.time()
        epoch_loss, epoch_accu = train(epoch, trainloader, model, device, optimizer, loss_function)
        print('---Train takes: %s seconds ---' %(time.time() - start_time))
        train_losses.append(epoch_loss)
        train_accu.append(epoch_accu)
        start_time = time.time()
        epoch_loss, epoch_accu = test(epoch, testloader, model, device, optimizer, loss_function)
        print('---Test takes: %s seconds ---' %(time.time() - start_time))
        eval_losses.append(epoch_loss)
        eval_accu.append(epoch_accu)

    # Clean memory
    del dataset
    del trainloader
    del testloader
 
    plt.plot(train_accu)
    plt.plot(eval_accu)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.savefig('carlos/data/plots/accuracy.png')
    plt.close()
 
    plt.plot(train_losses)
    plt.plot(eval_losses)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig('carlos/data/plots/losses.png')
    plt.close()

    print('Finished Training')

import csv
def testcsv():
    with open('logs/botbowl-11/5de641f4-7f9d-11ec-b508-ec63d79c77d6.dat', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row[1].strip())
 
if __name__ == "__main__":
    # create_dataset()
    # actions_stats()
    # main()
    testcsv()
 
        # running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
        #     # get the inputs; data is a list of [inputs, labels]
        #     spatial_obs, non_spatial_obs, actions = data
 
        #     # zero the parameter gradients
        #     optimizer.zero_grad()
 
        #     # forward + backward + optimize
        #     outputs = model(spatial_obs, non_spatial_obs)
        #     # print(spatial_obs.size(), non_spatial_obs.size(), actions.size(), outputs[1].size())
        #     # print(actions)
        #     loss = loss_function(outputs[1], actions)
        #     loss.backward()
        #     optimizer.step()
        #     print(loss.item())
        #     # print statistics
        #     running_loss += loss.item()
        #     if i % 2000 == 1999:    # print every 2000 mini-batches
        #         print('[%d, %5d] loss: %.3f' %
        #             (epoch + 1, i + 1, running_loss / 2000))
        #         running_loss = 0.0
 
 
 
 
 
 
    # m = nn.LogSoftmax(dim=1)
    # loss = nn.NLLLoss()
    # # input is of size N x C = 3 x 5
    # input = torch.randn(3, 5, requires_grad=True)
    # # each element in target has to have 0 <= value < C
    # target = torch.tensor([1, 0, 4])
    # output = loss(m(input), target)
    # print(m(input).size(), target.size())
    # output.backward()
    # # 2D loss example (used, for example, with image inputs)
    # N, C = 5, 4
    # loss = nn.NLLLoss()
    # # input is of size N x C x height x width
    # data = torch.randn(N, 16, 10, 10)
    # conv = nn.Conv2d(16, C, (3, 3))
    # m = nn.LogSoftmax(dim=1)
    # # each element in target has to have 0 <= value < C
    # target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    # print(m(conv(data)).size(), target.size())
    # output = loss(m(conv(data)), target)
    # output.backward()