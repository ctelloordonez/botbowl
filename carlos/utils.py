from fileinput import filename
import os
import torch
from agents.carlos_agent import CNNPolicy
from botbowl.ai.env import BotBowlEnv, EnvConf
import gym
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import time
import carlos
import numpy as np

def get_data_path(rel_path):
    root_dir = carlos.__file__.replace("__init__.py", "")
    filename = os.path.join(root_dir, "data/" + rel_path)
    return os.path.abspath(os.path.realpath(filename))
 
def create_dataset():
    pairs_directory = get_data_path('no_flip_pairs')
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
    filename = os.path.join(dataset_directory, f"no_flip_dataset.pt")
    torch.save(dataset, filename)


def get_action_type(action_idx, non_spatial_action_types, spatial_action_types, board_squares):
    if action_idx < len(non_spatial_action_types):
        return non_spatial_action_types[action_idx]
    spatial_idx = action_idx - len(non_spatial_action_types)
    spatial_action_type_idx = int(spatial_idx / board_squares)
    return spatial_action_types[spatial_action_type_idx]


def actions_stats(actions=None):
    if actions == None:
        dataset = load_dataset()
        actions = dataset['Y']
        del dataset
    env_conf = EnvConf(size=11, pathfinding=True)
    env = BotBowlEnv(env_conf=env_conf)

    action_types = env_conf.action_types
    actions_dic = {}
    for action_type in action_types:
        actions_dic[action_type.name] = 0
   
    positional_action_types = env_conf.positional_action_types
    positional_action_dic = {}
    for postional_action in positional_action_types:
        positional_action_dic[postional_action.name] = np.zeros((17, 28))

    for a, action in enumerate(actions):
        if a % 2000 == 0:
            print(f"{a}/{len(actions)}")
        action_idx = action.type(torch.IntTensor).item()
        action = env._compute_action(action_idx=action_idx, flip=False)[0]
        action_type = action.action_type.name

        count =actions_dic.get(action_type)
        if count == None:
            count = 0
        actions_dic.update({action_type : count+1})

        if not action.action_type in positional_action_types:
            continue
        x = action.position.x
        y = action.position.y
        positional_action_dic[action_type][y, x] += 1


    actions_percentages = {}
    for k, key in enumerate(actions_dic.keys()):
        type_count = actions_dic[key]
        actions_percentages[key] = (type_count / len(actions)) * 100
   
    for key in actions_percentages.keys():
        print("{}: {:.5f}".format(key, actions_percentages.get(key)))

    import matplotlib.pyplot as plt

    names = list(actions_percentages.keys())
    values = list(actions_percentages.values())

    fig, ax = plt.subplots()
    ax.bar(names, values)
    fig.suptitle('Dataset Train Distribution')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.show()

    fig, axs = plt.subplots(4, 5)
    x = 0
    y = 0
    for p, positional_action in enumerate(positional_action_dic.keys()):
        axs[x, y].imshow(positional_action_dic[positional_action])
        axs[x, y].set_title(positional_action)
        x += 1
        if x >= 4:
            x = 0
            y += 1
    fig.suptitle('Positional actions heatmap')
    plt.show()
 
def load_dataset():
    print('Loading dataset')
    directory = get_data_path('dataset')
    if not os.path.exists(directory):
        return
   
    filename = os.path.join(directory, f"no_flip_dataset.pt")
    dataset = torch.load(filename)
    # dataset['X_spatial'] = dataset['X_spatial'][0:400]
    # dataset['X_non_spatial'] = dataset['X_non_spatial'][0:400]
    # # dataset['action_masks'] = dataset['action_masks'][0:400]
    # dataset['Y'] = dataset['Y'][0:400]
    print('Dataset loaded')
    return dataset
   
 
def make_trainset(dataset):
    print('Making trainset')
    spatial_obs = torch.stack(dataset['X_spatial'][0:split])
    # print(spatial_obs.size())
    spatial_obs = torch.reshape(spatial_obs, (split, 44, 17, 28)) # TODO: fix magic number
    non_spatial_obs = torch.stack(dataset['X_non_spatial'][0:split])
    non_spatial_obs = torch.reshape(non_spatial_obs, (split, 1, 115))
    # action_masks = torch.stack(dataset['action_masks'][0:split])
    # action_masks = torch.reshape(action_masks, (split, 8117))
    # print(action_masks.shape)
    actions = torch.stack(dataset['Y'][0:split])
    actions = torch.flatten(actions)
    actions = actions.long()
 
    trainset = torch.utils.data.TensorDataset(spatial_obs, non_spatial_obs, actions)
    print('Trainset made')
    return trainset
 
def make_testset(dataset):
    print('Makig testset')
    spatial_obs = torch.stack(dataset['X_spatial'][split:-1])
    spatial_obs = torch.reshape(spatial_obs, (len(spatial_obs), 44, 17, 28)) # TODO: Fix magic number
    non_spatial_obs = torch.stack(dataset['X_non_spatial'][split:-1])
    non_spatial_obs = torch.reshape(non_spatial_obs, (len(non_spatial_obs), 1, 115))
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
        spatial_obs, non_spatial_obs, actions = data
        spatial_obs = spatial_obs.to(device)
        non_spatial_obs = non_spatial_obs.to(device)
        actions = actions.to(device)
 
        # zero the parameter gradients
        optimizer.zero_grad()
 
        # forward + backward + optimize
        values, action_log_probs, = model.get_action_log_probs(spatial_obs, non_spatial_obs)
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
 
    model_dir = get_data_path('no_flip_models')
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
        spatial_obs, non_spatial_obs, actions = data
        spatial_obs = spatial_obs.to(device)
        non_spatial_obs = non_spatial_obs.to(device)
        actions = actions.to(device)
 
        # zero the parameter gradients
        optimizer.zero_grad()
 
        # forward + backward + optimize
        values, action_log_probs, = model.get_action_log_probs(spatial_obs, non_spatial_obs)
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
   
 
    env_conf = EnvConf(size=11, pathfinding=True)
    env = BotBowlEnv(env_conf=env_conf)
    env.reset()
    spat_obs, non_spat_obs, action_mask = env.get_state()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)

    # state_dict_file = torch.load("carlos/data/no_flip_models/epoch-21.pth")
    model = CNNPolicy(spatial_obs_space, non_spatial_obs_space, hidden_nodes=num_hidden_nodes, kernels=num_cnn_kernels, actions=action_space)
    # model.load_state_dict(state_dict_file['model_state_dict'])
   
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

        # Reload models
        directory = get_data_path('no_flip_models')
        filename = os.path.join(directory, f"epoch-{epoch}.pth")
        state_dict_file = torch.load(filename)
        model.load_state_dict(state_dict_file['model_state_dict'])
        optimizer.load_state_dict(state_dict_file['optimizer_state_dict'])

        epoch_loss = state_dict_file['loss']
        epoch_accu = state_dict_file['accuracy']
        print('Train Loss: %.3f | Accuracy: %.3f'%(epoch_loss, epoch_accu))

        # epoch_loss, epoch_accu = train(epoch, trainloader, model, device, optimizer, loss_function)
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
 
    plots_dir = get_data_path('no_flip_plots')
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    plt.plot(train_accu)
    plt.plot(eval_accu)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.savefig(os.path.join(plots_dir, 'accuracy.png'))
    plt.close()
 
    plt.plot(train_losses)
    plt.plot(eval_losses)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig(os.path.join(plots_dir, 'losses.png'))
    plt.close()

    print('Finished Training')
 
if __name__ == "__main__":
    # create_dataset()
    # main()
    actions_stats()