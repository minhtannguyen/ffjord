
import os, sys, argparse, logging

import torch
from data_provider import datasets_factory
# from convlstmnet import ConvLSTMNet, ConvTTLSTMNet
from convlstmnet_test import ConvLSTMNet12, ConvLSTMNet20, ConvLSTMNet_
# from ffmpeg_gif import save_gif

import torchvision
import skimage
import numpy

def main(args):
    # Data preparation (Moving-MNIST dataset)
    DATA_DIR   = args.data_path

    batch_size  = args.batch_size
    log_samples = args.log_samples
    assert log_samples % batch_size == 0, \
        "log_samples should be a multiple of batch_size."

    test_data_paths = os.path.join(DATA_DIR, "moving-mnist-test.npz")
    test_input_handle = datasets_factory.data_provider(
        dataset_name = "mnist", batch_size = batch_size, img_width = 64, 
        train_data_paths = None, valid_data_paths = None, test_data_paths = test_data_paths,
        is_training = False, return_test = True)
    test_size = test_input_handle.total()

    # split of frames
    input_frames  = args.input_frames
    future_frames = args.future_frames
    output_frames = args.output_frames

    total_frames = 20 # returned by the data provider
    train_frames  = input_frames + future_frames
    assert train_frames <= total_frames, \
        "The number of train_frames(input_frames + future_frames) should be less than total_frames(20)."

    # Model preparation (Conv-LSTM network)
    if args.model_size == 'large': # 20-layers
        hidden_channels  = (64, 128, 128, 64)
        layers_per_block = (5, 5, 5, 5)
        skip_stride = 2
    elif args.model_size == "small": # 12-layers
        hidden_channels  = (32, 48, 48, 32)
        layers_per_block = (3, 3, 3, 3)
        skip_stride = 2
    elif args.model_size == "test": # 4-layers
        hidden_channels  = (128, )
        layers_per_block = (4, )
        skip_stride = None 
    else:
        raise NotImplementedError

    model = ConvLSTMNet(
        # model architecture
        layers_per_block = layers_per_block, hidden_channels = hidden_channels, 
        input_channels = 1, skip_stride = skip_stride,
        # computational module
        cell = args.model, kernel_size = 3, bias = True,
        # input and output formats
        input_frames = input_frames, future_frames = future_frames, 
        output_sigmoid = False, output_concat = True)

    # whether to use GPU for training
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)

    # whether to use multi-GPU for training
    if use_cuda and args.multi_gpu and torch.cuda.device_count() > 1:
        print("Number of GPUs: ", torch.cuda.device_count())
        model = nn.DataParallel(model)

    model.to(device)

    # load the model from the specified epoch
    MODEL_DIR = os.path.join(args.model_path, args.model_name)
    eval_epoch = args.eval_epoch
    MODEL_FILE = os.path.join(args.model_path, args.model, 'training_%d.pt' % eval_epoch)
    model.load_state_dict(torch.load(MODEL_FILE))

    RESULT_DIR = os.path.join(args.result_path, args.model_name)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    ## Evaluation on the test set
    MSE, PSNR, SSIM = 0.0, 0.0, 0.0 # statistics
    with torch.no_grad():
        batch = 0
        while not test_input_handle.no_batch_left():
            inputs = torch.from_numpy(test_input_handle.get_batch())
            test_input_handle.next()
            batch += batch_size

            # 5-th order: batch_size(0) x frames(1) x input_channels(2) x height(3) x width(4) 
            inputs = inputs.permute(0, 1, 4, 2, 3).to(device)
            # 5-th order: batch_size(0) x output_frames(1) x input_channels(2) x height(3) x width(4)
            origin = inputs[:, input_frames:]
            # 5-th order: batch_size(0) x input_frames(1) x input_channels(2) x height(3) x width(4)
            inputs = inputs[:, :input_frames]

            # 5-th order: batch_size(0) x output_frames(1) x input_channels(2) x height(3) x width(4)
            pred = model(inputs)

            if batch % 100 == 0:
                print('Testing: {}/{}'.format(batch, test_input_handle.total()))
                origin_ = torch.split(origin, 1, dim = 1)
                pred_   = torch.split(pred,   1, dim = 1)

                for t in range(output_frames):
                    # 4-th order: batch_size(0) x input_channels(1) x height(2) x width (3) 
                    origin_[t] = torch.squeeze(origin_[t], dim = 1)
                    pred_[t] = torch.squeeze(pred_[t], dim = 1)
                    torchvision.utils.save_image(torch.cat([origin_[t], pred_[t]], dim = 0), 
                        os.path.join(RESULT_DIR, "comparison_%d_%d.jpg" % (batch, t+1)), nrow = batch_size)

                # frames[t] = frames[t].permute(1, 2, 0).cpu().numpy()
                # save_gif(os.path.join(RESULT_DIR, "sample_%d.gif" % batch), frames, fps = 4)

            # 2) accumulate the statistics
            origin = torch.squeeze(origin, dim = 2)
            origin = origin.cpu().numpy()
            pred = torch.squeeze(pred, dim = 2)
            pred = pred.cpu().numpy()
            for i in range(batch_size):
                for t in range(output_frames):
                    MSE  += skimage.measure.compare_mse(origin[i, t], pred[i, t])
                    PSNR += skimage.measure.compare_psnr(origin[i, t], pred[i, t])
                    SSIM += skimage.measure.compare_ssim(origin[i, t], pred[i, t])

        print("MSE:  ", 1e3 * MSE  / (test_input_handle.total() * output_frames), "(x 1e-3)")
        print("PSNR: ", PSNR / (test_input_handle.total() * output_frames))
        print("SSIM: ", SSIM / (test_input_handle.total() * output_frames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Moving-MNIST ConvLSTM Training")

    # basics
    parser.add_argument('--model', default = 'convlstm', type = str,
        help = 'The model is either convlstm or convttlstm.')
    parser.add_argument('--use-cuda', default = True, type = bool,
        help = 'Whether to use GPU for testing.')
    parser.add_argument('--batch_size', default = 4, type = int,
        help = 'The batch size in testing/display.')

    # paths
    parser.add_argument('--model-path', default = '../mnist_models', type = str,
        help = "The path to the folder storing the models.")
    parser.add_argument('--data-path', default = '../datasets/moving-mnist', type = str,
        help = 'The path to the folder storing the dataset.')
    parser.add_argument('--result-path', default = '../mnist_results', type = str,
        help = 'The path to the folder storing the results.')

    # epoch
    parser.add_argument('--eval-epoch', default = 20, type = int,
        help = 'Evaluate the model at the specified epoch.')

    args = parser.parse_args()
    main(args)