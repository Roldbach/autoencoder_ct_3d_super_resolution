"""Script for model training.

This is the main script used for model training.
"""
import time

import torch

from configuration import device_configuration
from helper import train_helper
from utils import io_utils
from utils import setup_utils


if __name__ == '__main__':
    model_name = 'PlainCNN'
    upsample_name = 'trilinear_interpolation'
    scale_factor = 8
    weight_path = f'./temp/{model_name}_{upsample_name}_x{scale_factor}.pth'
    record_path = f'./temp/record.csv'
    require_loading = False

    window = (-1024, 1476)
    patch_size = 64
    epoch = 500
    batch_size = 16
    learning_rate = 1e-5
    train_data_loader = setup_utils.set_up_data_loader('train')
    validation_data_loader = setup_utils.set_up_data_loader('validation')
    model = setup_utils.set_up_model(
        model_name,
        device_configuration.TRAIN_DEVICE,
        device_configuration.TRAIN_DEVICE_ID,
    )
    optimiser = setup_utils.set_up_optimiser(model.parameters(), learning_rate)
    loss_function = setup_utils.set_up_loss_function()
    record = setup_utils.set_up_record()

    if require_loading:
        model = io_utils.read_weight(weight_path, model)
        record = io_utils.read_file(record_path)
    else:
        train_helper.reset_directory('./temp')

    while len(record) < epoch:
        validation_batch_loss_accumulator = train_helper.BatchLossAccumulator()
        train_batch_loss_accumulator = train_helper.BatchLossAccumulator()

        time_start = time.time()

        with torch.no_grad():
            model.eval()
            for input, label in validation_data_loader:
                train_helper.validate_epoch(
                    input, label,
                    window, scale_factor, upsample_name, patch_size, batch_size,
                    model, loss_function, validation_batch_loss_accumulator,
                )
        
        model.train()
        for input, label in train_data_loader:
            train_helper.train_epoch(
                input, label,
                window, scale_factor, upsample_name, patch_size, batch_size,
                model, optimiser, loss_function, train_batch_loss_accumulator,
            )

        time_end = time.time()

        epoch_data = {
            'epoch': len(record) + 1,
            'train_loss': train_batch_loss_accumulator.average_batch_loss(),
            'validation_loss':
                validation_batch_loss_accumulator.average_batch_loss(),
            'time': time_end - time_start,
        }
        record = train_helper.append_record(epoch_data, record)

        io_utils.write_weight(weight_path, model)
        io_utils.write_file(record_path, record)

        train_helper.report_epoch_data(epoch_data)

        torch.cuda.empty_cache()