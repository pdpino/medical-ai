import torch
# from torch.nn import DataParallel
# import os

# from cxr8 import utils
# from cxr8.training import optimizers
from mrg.models.classification import resnet

_MODELS_DEF = {
    'resnet': resnet.Resnet50CNN,
}

AVAILABLE_MODELS = list(_MODELS_DEF)


def init_empty_model(model_name, labels, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')
    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(labels, **kwargs)


    # model = DataParallel(model) # Always set to use multiple gpus

    return model


# def get_model_fname(run_name, experiment_mode="debug", base_dir=utils.BASE_DIR):
#     folder = os.path.join(base_dir, "models")
#     if experiment_mode:
#         folder = os.path.join(folder, experiment_mode)
#     os.makedirs(folder, exist_ok=True)
#     return os.path.join(folder, run_name + ".pth")


# def save_model(run_name, model_name, experiment_mode, hparam_dict, trainer, model, optimizer,
#                base_dir=utils.BASE_DIR,
#                ):
#     model_fname = get_model_fname(run_name, experiment_mode=experiment_mode, base_dir=base_dir)
#     torch.save({
#         "model_name": model_name,
#         "hparams": hparam_dict,
#         "epoch": trainer.state.epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#     }, model_fname)


# def load_model(run_name, experiment_mode="", device=None,
#                force_multiple_gpu=False,
#                base_dir=utils.BASE_DIR):
#     model_fname = get_model_fname(run_name, experiment_mode=experiment_mode, base_dir=base_dir)
    
#     checkpoint = torch.load(model_fname, map_location=device)
#     hparams = checkpoint["hparams"]
#     model_name = checkpoint.get("model_name", "v0")
#     chosen_diseases = hparams["diseases"].split(",")
#     train_resnet = hparams["train_resnet"]
#     multiple_gpu = hparams.get("multiple_gpu", False)
    
#     def extract_params(name):
#         params = {}
#         prefix = name + "_"
#         for key, value in hparams.items():
#             if key.startswith(prefix):
#                 key = key[len(prefix):]
#                 params[key] = value
#         return params
    
#     opt_params = extract_params("opt")

#     # Load model
#     model = init_empty_model(model_name, chosen_diseases, train_resnet)
    
#     # NOTE: this force param has to be used for cases when the hparam was not saved
#     if force_multiple_gpu or multiple_gpu:
#         model = DataParallel(model)
    
#     if device:
#         model = model.to(device)
    
#     # Load optimizer
#     opt_name = hparams["opt"]
#     OptClass = optimizers.get_optimizer_class(opt_name)
#     optimizer = OptClass(model.parameters(), **opt_params)

#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#     # Load loss
#     loss_name = hparams["loss"]
#     loss_params = extract_params("loss")
    
#     # TODO: make a CompiledModel class to hold all of these values (and avoid changing a lot of code after any change here)
#     return model, model_name, optimizer, opt_name, loss_name, loss_params, chosen_diseases
