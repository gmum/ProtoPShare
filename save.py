import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print, joint_optimizer=None, warm_optimizer=None, last_layer_optimizer=None):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save({'joint_optimizer': joint_optimizer,
                'warm_optimizer': warm_optimizer,
                'last_layer_optimizer': last_layer_optimizer},
                os.path.join(model_dir, (model_name + '{0:.4f}_optims.pth').format(accu)))

