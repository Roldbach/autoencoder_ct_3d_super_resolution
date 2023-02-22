"""Script for model testing.

This is the main script used for model testing.
"""
from configuration import device_configuration
from helper import test_helper
from utils import io_utils
from utils import setup_utils

import matplotlib.pyplot as plt


def plotImage(image, name, path):
    figure=plt.figure(dpi=300)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(image,cmap=plt.cm.gray)
    figure.savefig(path+"/"+name+".png")

if __name__ == '__main__':
    model_name = 'PlainCNN'
    upsample_name = 'trilinear_interpolation'
    scale_factor = 4
    weight_path = f'./weight/{model_name}_{upsample_name}_x{scale_factor}.pth'
    window = (-1024, 1476)

    test_data_loader = setup_utils.set_up_data_loader('test', shuffle=False)
    model = setup_utils.set_up_model(model_name, device_configuration.TEST_DEVICE)
    model = io_utils.read_weight(weight_path, model)

    input_vs_label, prediction_vs_label = [], []

    count = 0
    model.eval()
    for input, label in test_data_loader:
        input = test_helper.pre_process_input(
            input, window, scale_factor, upsample_name)
        label = test_helper.pre_process_label(label, window, scale_factor)

        prediction = model(input.to(device_configuration.TEST_DEVICE))

        input = test_helper.post_process_image(input)
        prediction = test_helper.post_process_image(prediction)
        label = test_helper.post_process_image(label)

        input_vs_label.append(test_helper.evaluate(input, label))
        prediction_vs_label.append(test_helper.evaluate(prediction, label))

        if count == 0:
            plotImage(input[:, :, 100], f'input_{count}', '.')
            plotImage(prediction[:, :, 100], f'prediction_{count}', '.')
            plotImage(label[:, :, 100], f'label_{count}', '.')

    test_helper.report_evaluation_result(input_vs_label, prediction_vs_label)