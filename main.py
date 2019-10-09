from MoE.lib.heter_utils import gen_heter_data, wrap_heter_data
import numpy as np
import string, copy, random
from MoE.lib.model import MoO, AdaptiveGate, MLP
import torch, tqdm
from MoE.lib.heter_utils import train, test, gen_heter_data, moving_average
import argparse
from sklearn.externals import joblib
from collections import deque

def get_args():
    parser=argparse.ArgumentParser() 
    parser.add_argument('-d', type=int, default=30)
    parser.add_argument('-n', type=int, default=3000)
    parser.add_argument('--n_groups', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=800)
    parser.add_argument('--cycle', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--collect_ensemble_from', type=float, default=0.8)
    parser.add_argument('--learn_gate_from', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=100)     
    parser.add_argument('--savename', type=str, default="name")    
    args=parser.parse_args()
    return args

def random_string(N=5):  
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(N))

def save_run(args):
    joblib.dump(args, args['savename'] + "_" + random_string() + ".pkl") 

def train_ensemble(net, train_loader, val_loader, test_loader, criterion,
                   opt, n_epochs, cycle, collect_ensemble_from,
                   patience, args): # patience = max(10, 0.1 * B)

    # determine the budget to train a single model with early stopping
    experts = deque() # collect the last few snapshots of models
    B = 0
    train_losses, val_losses, test_losses = [], [], []
    val_loss_min, not_been_updated = np.infty, 0
    for i in tqdm.tqdm(range(n_epochs)):
        train(net, train_loader, criterion, opt, 1)
        B += 1
        
        train_losses.append(test(net, train_loader, criterion))
        val_losses.append(test(net, val_loader, criterion))
        test_losses.append(test(net, test_loader, criterion))

        if val_losses[-1] <= val_loss_min:
            not_been_updated = 0
            val_loss_min = val_losses[-1]
        else:
            not_been_updated += 1
            if not_been_updated > patience:
                break

        if i % cycle == 0:
            experts.append((i, copy.deepcopy(net)))
        while len(experts) > 0 and experts[0][0] < int(collect_ensemble_from * B):
            experts.popleft()

    min_index = np.argmin(val_losses)
    print('test loss: {:.5f}, train loss: {:.5f}'.format(test_losses[min_index],
                                                         train_losses[min_index]))

    forward_function = lambda x: torch.softmax(torch.ones(x.shape[0], len(experts)),
                                               dim=1)
    gate = AdaptiveGate(args.d, len(experts), forward_function=forward_function)
    net_ensemble = MoO([ex for i, ex in experts], gate) 
    test_loss_ensemble = test(net_ensemble, test_loader, criterion)
    train_loss_ensemble = test(net_ensemble, train_loader, criterion)
    print('test FGE: {:.5f}, train FGE: {:.5f}'.format(test_loss_ensemble,
                                                       train_loss_ensemble))
    return train_losses, val_losses, test_losses, experts, B, \
        train_loss_ensemble, test_loss_ensemble, train_losses[min_index], \
        test_losses[min_index]

def train_gate(net, train_loader, val_loader, test_loader, criterion,
               opt, n_epochs, cycle, patience):
    train_losses, val_losses, test_losses = [], [], []
    val_loss_min, not_been_updated = np.infty, 0
    for i in tqdm.tqdm(range(n_epochs)):
        train(net, train_loader, criterion, opt, 1)

        train_losses.append(test(net, train_loader, criterion))
        val_losses.append(test(net, val_loader, criterion))
        test_losses.append(test(net, test_loader, criterion))
        
        if val_losses[-1] <= val_loss_min:
            not_been_updated = 0
            val_loss_min = val_losses[-1]
        else:
            not_been_updated += 1
            if not_been_updated > patience:
                break

    min_index = np.argmin(val_losses)
    print('test loss MoE: {:.5f}, train loss MoE: {:.5f}'.format(test_losses[min_index],
                                                                 train_losses[min_index])
    )
    return train_losses, val_losses, test_losses, train_losses[min_index], \
        test_losses[min_index]

def main():
    args = get_args()
    torch.set_num_threads(1)

    # dataset
    x, y, r, z = gen_heter_data(args.n, args.d, args.n_groups)
    n = args.n
    train_loader = wrap_heter_data(x[:n//3], y[:n//3], bs=args.batch_size)
    val_loader = wrap_heter_data(x[n//3:n*2//3], y[n//3: n*2//3])
    test_loader = wrap_heter_data(x[n*2//3:], y[n*2//3:])

    # get model
    net = MLP([args.d, args.d, args.d, 2])

    # training with early stopping
    print('training...')
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    train_losses, val_losses, test_losses, experts, B, \
        train_loss_ensemble, test_loss_ensemble, train_loss_single, \
        test_loss_single = train_ensemble(
        net, train_loader, val_loader, test_loader, criterion, opt,
        args.n_epochs, args.cycle, args.collect_ensemble_from, args.patience, args)

    # train the gating
    print('training gate...')
    experts_moe = [ex for i, ex in experts if i <= args.learn_gate_from * B]
    assert len(experts_moe) >= 1, "expert length >= 1"
    print('{} experts'.format(len(experts_moe)))
    forward_function = MLP([args.d, args.d, args.d, len(experts_moe)])
    gate = AdaptiveGate(args.d, len(experts_moe), forward_function=forward_function)
    net_moe = MoO(experts_moe, gate)
    opt = torch.optim.Adam(forward_function.parameters(), lr=args.lr)
    gate_train_losses, gate_val_losses, gate_test_losses, \
        train_loss_moe, test_loss_moe = train_gate(
        net_moe, train_loader, val_loader, test_loader, criterion, opt,
        int((1-args.learn_gate_from) * B), args.cycle, args.patience, )
        
    # SWA: todo
    
    # saving: todo: save all experts
    args = vars(args) # turn into dict
    args['x'] = x
    args['y'] = y
    args['r'] = r
    args['z'] = z
    args['test_losses'] = test_losses
    args['train_losses'] = train_losses
    args['val_losses'] = val_losses
    args['gate_test_losses'] = gate_test_losses
    args['gate_train_losses'] = gate_train_losses
    args['gate_val_losses'] = gate_val_losses
    args['train_loss_single'] = train_loss_single
    args['test_loss_single'] = test_loss_single
    args['train_loss_ensemble'] = train_loss_ensemble
    args['test_loss_ensemble'] = test_loss_ensemble    
    args['train_loss_moe'] = train_loss_moe
    args['test_loss_moe'] = test_loss_moe
    
    save_run(args)

if __name__ == '__main__':
    main()
