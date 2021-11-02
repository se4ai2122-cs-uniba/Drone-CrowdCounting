import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from models.CC import CrowdCounter
from ruamel.yaml import YAML
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataset.visdrone import load_test, cfg_data
from config import cfg

losses_dict = {
    'rmse': lambda x, y: mean_squared_error(x, y, squared=False),
    'mae': mean_absolute_error
    }


def load_CC_test():

    """
    Load CrowdCounter model net for testing mode
    """
    cc = CrowdCounter([0], cfg.NET)
    if cfg.PRE_TRAINED:
        cc.load(cfg.PRE_TRAINED)
    return cc

def test_model(
        model,
        data_test,
        batch_size=128,
        n_workers=4,
        device=None,
        out_prediction=None
):
    """
    Test the given model on a given dataset

    @param model: torch model to test
    @param data_test: torch dataset for testing the madel
    @param batch_size: batch size for parallel computation
    @param n_workers: n° workers for parallel processing
    @param device: device where to compute the network calculations (cuda or cpu)
    @param out_prediction: boolean that specify if saving the heatmaps generated
    @return: y_true and y_pres
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Make sure the model is set to evaluation mode
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, (*inputs, targets) in enumerate(tqdm(test_loader)):
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)
            predictions = model.predict(*inputs)

            count_pr = torch.sum(predictions.squeeze(), dim=(1, 2)) / 2550
            count_gt = torch.sum(targets.squeeze(), dim=(1, 2))

            if out_prediction:
                for img in range(predictions.shape[0]):
                    out_dir = os.path.join(os.path.dirname(out_prediction), 'preds')
                    Path(out_dir).mkdir(exist_ok=True)
                    plt.imsave(
                        os.path.join(out_dir, str(i)) + '.png',
                        predictions[img].squeeze().cpu().numpy(),
                        cmap='jet', )

            y_pred.extend(count_pr.cpu().tolist())
            y_true.extend(count_gt.cpu().tolist())

        return y_true, y_pred


def evaluate_model(model_function, data_function, bs, n_workers, losses, device=None, out_prediction=None):
    """
    Evaluate a given model on a given dataset using the given loss functions

    @param model_function: function that returns the torch model
    @param data_function: function that return the torch dataset
    @param bs: batch size for parallel computation
    @param n_workers: n° workers for parallel processing
    @param losses: list of loss functions
    @param device: device where to compute the network calculations (cuda or cpu)
    @param out_prediction: boolean that specify if saving the heatmaps generated
    @return: list of loss values
    """
    ds = data_function()
    net = model_function()
    net = net.to(device)
    y_true, y_pred = test_model(net, ds, bs, n_workers, device, out_prediction)

    results = {}
    for loss in losses:
        results[loss] = losses[loss](y_pred, y_true)

    return results


if __name__ == '__main__':

    params_path = Path("../params.yaml")
    with open(params_path, 'r') as params_file:
        yaml = YAML()
        params = yaml.load(params_file)

        eval_params = params['evaluate']
        global_params = params['global']
        cfg.NET, cfg.GPU = eval_params['model']['NET'], eval_params['model']['GPU']
        cfg.PRE_TRAINED = eval_params['model']['PRETRAINED']
        cfg.N_WORKERS = eval_params['N_WORKERS']
        cfg.TEST_BATCH_SIZE = eval_params['BATCH_SIZE']
        cfg.DF_PATH = os.path.join(global_params['DATA_PATH'], 'test')
        cfg.DEVICE = eval_params['DEVICE']
        cfg.OUT_PREDICTIONS = eval_params['OUT_PREDICTIONS']
        cfg.LOSSES = eval_params['LOSSES']
        cfg_data.SIZE = global_params['SIZE']

        actual_losses = cfg.LOSSES

        losses = {loss: losses_dict[loss] for loss in actual_losses}

        print(evaluate_model(load_CC_test, load_test, cfg.TEST_BATCH_SIZE, cfg.N_WORKERS, losses, cfg.DEVICE, cfg.OUT_PREDICTIONS))