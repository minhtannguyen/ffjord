import os, sys, argparse, logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import skimage
from tensorboardX import SummaryWriter 

from data_provider import datasets_factory

# from convlstmnet import ConvLSTMNet
# from convlstmnet_test import ConvLSTMNet12, ConvLSTMNet20, ConvLSTMNet_

def main(args):
    ## Data preparation (Moving-MNIST dataset)
    DATA_DIR    = args.data_path

    batch_size  = args.batch_size
    log_samples = args.log_samples
    assert log_samples % batch_size == 0, \
        "log_samples should be a multiple of batch_size."
    
    train_data_paths = os.path.join(DATA_DIR, "moving-mnist-train.npz")
    valid_data_paths = os.path.join(DATA_DIR, "moving-mnist-valid.npz")

    train_input_handle, valid_input_handle = datasets_factory.data_provider(
        dataset_name = "mnist", batch_size = batch_size, img_width = 64, 
        train_data_paths = train_data_paths, valid_data_paths = valid_data_paths, test_data_paths = None,
        is_training = True, return_test = False)
    train_size = train_input_handle.total()
    valid_size = valid_input_handle.total()

    # split of frames
    input_frames  = args.input_frames
    future_frames = args.future_frames
    output_frames = args.output_frames

    total_frames = 20 # returned by the data provider
    train_frames  = input_frames + future_frames
    assert train_frames <= total_frames, \
        "The number of train_frames(input_frames + future_frames) should be less than total_frames(20)."

#     # Model preparation (Conv-LSTM network)
#     if args.model_size == 'large': # 20-layers
#         hidden_channels  = (64, 128, 128, 64)
#         layers_per_block = (5, 5, 5, 5)
#         skip_stride = 2
#     elif args.model_size == "small": # 12-layers
#         hidden_channels  = (32, 48, 48, 32)
#         layers_per_block = (3, 3, 3, 3)
#         skip_stride = 2
#     elif args.model_size == "test": # 4-layers
#         hidden_channels  = (128, )
#         layers_per_block = (4, )
#         skip_stride = None 
#     else:
#         raise NotImplementedError

#     assert args.model in ["convlstm", "convttlstm"], \
#         "The specified model is not currently supported."

#     model = ConvLSTMNet(
#         # model architecture
#         layers_per_block = layers_per_block, hidden_channels = hidden_channels, 
#         input_channels = 1, skip_stride = skip_stride,
#         # computational module
#         cell = args.model, kernel_size = 3, bias = True,
#         # input and output formats
#         input_frames = input_frames, future_frames = future_frames, 
#         output_sigmoid = False, output_concat = True)

    # if args.model_size == "small": 
    #     model = ConvLSTMNet12(channels = 1, 
    #         input_frames = input_frames, future_frames = future_frames,
    #         output_sigmoid = False, output_concat = True)
    # elif args.model_size == "large":
    #     model = ConvLSTMNet20(channels = 1, 
    #         input_frames = input_frames, future_frames = future_frames, 
    #         output_sigmoid = False, output_concat = True)
    # elif args.model_size == "test":
    #     model = ConvLSTMNet4(channels = 1,
    #         input_frames = input_frames, future_frames = future_frames,
    #         output_sigmoid = False, output_concat = True)
    # else:
    #     raise NotImplementedError

#     # whether to use GPU for training
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
#     print("device: ", device)

#     # whether to use multi-GPU for training
#     if use_cuda and args.multi_gpu and torch.cuda.device_count() > 1:
#         print("Number of GPUs: ", torch.cuda.device_count())
#         model = nn.DataParallel(model)

#     model.to(device)

#     if args.loss_function == "l1":
#         loss_func = lambda pred, origin: F.l1_loss(pred, origin, reduction = "mean")
#     elif args.loss_function == "l2":
#         loss_func = lambda pred, origin: F.mse_loss(pred, origin, reduction = "mean")
#     elif args.loss_function == "l1l2":
#         loss_func = lambda pred, origin: F.l1_loss(pred, origin, reduction = "mean") + \
#             F.mse_loss(pred, origin, reduction = "mean")
#     else:
#         raise NotImplementedError

#     teacher_forcing = args.teacher_forcing 

#     # load the model parameters from the specified epoch
#     MODEL_DIR = os.path.join(args.model_path, args.model_name)
    start_epoch = args.start_epoch
#     if start_epoch == 0:
#         if not os.path.exists(MODEL_DIR):
#             os.makedirs(MODEL_DIR)
#         MODEL_FILE = os.path.join(MODEL_DIR, 'training_init.pt')
#         torch.save(model.state_dict(), MODEL_FILE)
#     else: # if start_epoch > 0:
#         MODEL_FILE = os.path.join(MODEL_DIR, "training_%d.pt" % start_epoch)
#         assert os.path.exists(MODEL_FILE), "The model file does not exist."
#         model.load_state_dict(torch.load(MODEL_FILE))

#     ## Result destination
#     RESULT_DIR = os.path.join(args.result_path, args.model_name)
#     if not os.path.exists(RESULT_DIR):
#         os.makedirs(RESULT_DIR)
#     RESULT_FILE = os.path.join(RESULT_DIR, args.log_filename)
#     tensorboard_writer = SummaryWriter(RESULT_FILE)

    ## Hyperparameters of learning scheduling

    # learning rate (the rate is decayed exponentially)
    lr_decay_epoch = args.lr_decay_epoch 
    lr_decay_rate  = args.lr_decay_rate
    learning_rate  = args.learning_rate * (lr_decay_rate ** (start_epoch // lr_decay_epoch))
    # optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # scheduled sampling ratio (the ratio is decayed linearly)
    teacher_forcing = args.teacher_forcing
    if teacher_forcing:
        ssr_start_epoch = args.ssr_start_epoch
        ssr_decay_epoch = args.ssr_decay_epoch
        ssr_decay_ratio = args.ssr_decay_ratio
        scheduled_sampling_ratio = max(1 - ssr_decay_ratio * 
            (max(start_epoch - ssr_start_epoch, 0) // ssr_decay_epoch), 0)
    else: # if not teacher_forcing:
        scheduled_sampling_ratio = 0

    num_epochs = args.num_epochs  # total num of training epochs
    save_epoch = args.save_epoch  # save the model per save_epoch 

    total_samples = start_epoch * train_size
    for epoch in range(start_epoch, num_epochs):
#         # log the hyperparameters of learning sheduling 
#         tensorboard_writer.add_scalar('lr',  optimizer.param_groups[0]['lr'], epoch + 1)
#         tensorboard_writer.add_scalar('ssr', scheduled_sampling_ratio, epoch + 1)

        # Phase 1: Learning on the training set
        samples, LOSS = 0, 0.
        train_input_handle.begin(do_shuffle = True)
        while not train_input_handle.no_batch_left():
            # 5-th order: batch_size x total_frames x channels x height x width 
            frames = torch.from_numpy(train_input_handle.get_batch())
            import ipdb; ipdb.set_trace()
            frames = frames.permute(0, 1, 4, 2, 3).to(device)
            train_input_handle.next()

            if teacher_forcing:
                # frames = train_frames - 1 (0 ~ train_frames - 2)
                inputs = frames[:, :-1]
            else: # if not teacher_forcing:
                # frames = input_frames (0 ~ input_frames - 1)
                inputs = frames[:, :input_frames] 

            # frames = output_frames (train_frames - output_frames + 1 ~ train_frames)
            import ipdb; ipdb.set_trace()
            origin = frames[:, -output_frames:] 
            pred = model(inputs, output_frames = output_frames, 
                teacher_forcing = teacher_forcing, scheduled_sampling_ratio = scheduled_sampling_ratio)

            loss = loss_func(pred, origin)

            # accumulate the losses
            total_samples += batch_size 
            samples += batch_size
            LOSS += loss.item() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if samples % log_samples == 0:
                LOSS /= log_samples
                print('Epoch: {}/{}, Training: {}/{}, Loss: {}'.format(
                    epoch + 1, num_epochs, samples, train_size, LOSS))
                tensorboard_writer.add_scalar('LOSS', LOSS, total_samples)
                LOSS = 0.

        # adjust the learning rate of the optimizer 
        if (epoch + 1) % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_rate

        # adjust the scheduled sampling ratio
        if epoch >= ssr_start_epoch and (epoch + 1) % ssr_decay_epoch:
            scheduled_sampling_ratio = max(scheduled_sampling_ratio - ssr_decay_ratio, 0) 

        # save the model every save_epoch epochs
        if (epoch + 1) % save_epoch == 0:  
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'training_%d.pt' % (epoch + 1)))

        # Phase 2: evaluation on the validation set
        samples, LOSS = 0, 0.  
        MSE  = [0.] * future_frames
        PSNR = [0.] * future_frames
        SSIM = [0.] * future_frames

        with torch.no_grad():
            valid_input_handle.begin(do_shuffle = False)
            while not valid_input_handle.no_batch_left():
                # 5-th order: batch_size x total_frames x channels x height x width 
                frames = torch.from_numpy(valid_input_handle.get_batch())
                frames = frames.permute(0, 1, 4, 2, 3).to(device)
                valid_input_handle.next()

                # frames = input_frames (start_frames ~ start_frames + input_frames - 1)
                inputs = frames[:, :input_frames]

                # frames = future_frames (start_frames + train_frames - future_frames + 1 ~ start_frames + train_frames)
                origin = frames[:, -future_frames:]
                pred = model(inputs, output_frames = future_frames, teacher_forcing = False)

                loss = loss_func(pred, origin)
                pred = torch.clamp(pred, 0, 1)

                # accumulate the losses
                samples += batch_size
                LOSS += loss.item() * batch_size

                # display the sequence of inputs/ground-truth/predictions
                if samples % log_samples == 0:
                    print('Epoch: {}/{}, Validation: {}/{}'.format(
                        epoch + 1, num_epochs, samples, valid_size))

                    if input_frames >= future_frames:
                        input_0 = inputs[0, -future_frames:]
                    else:
                        input_0 = torch.cat([torch.zeros(future_frames - input_frames, 
                            1, 64, 64, device = device), inputs[0]], dim = 1)

                    img = torchvision.utils.make_grid(torch.cat([input_0, 
                        origin[0], pred[0]], dim = 0), nrow = future_frames)

                    RESULT_FILE = os.path.join(RESULT_DIR, "seq_%d_%d.jpg" % (samples, t+1))
                    torchvision.utils.save_image(img, RESULT_FILE)
                    tensorboard_writer.add_image("img_results", img, epoch + 1)

                # squeeze the channels axis, as MNIST is a gray image.
                # 4-th order: batch_size x future_frames x height x width
                origin = torch.squeeze(origin, dim = -3).cpu().numpy()
                pred = torch.squeeze(pred, dim = -3).cpu().numpy()
                for i in range(batch_size):
                    for t in range(future_frames):
                        MSE[t]  += skimage.measure.compare_mse(origin[i, t],  pred[i, t])
                        PSNR[t] += skimage.measure.compare_psnr(origin[i, t], pred[i, t])
                        SSIM[t] += skimage.measure.compare_ssim(origin[i, t], pred[i, t])

        LOSS /= valid_size
        tensorboard_writer.add_scalar("LOSS(val)", LOSS, epoch + 1)

        for t in range(future_frames):
            MSE[t]  /= valid_size
            PSNR[t] /= valid_size
            SSIM[t] /= valid_size
            tensorboard_writer.add_scalar("MSE_%d"  % (t + 1), MSE[t], epoch + 1)
            tensorboard_writer.add_scalar("PSNR_%d" % (t + 1), PSNR[t], epoch + 1)
            tensorboard_writer.add_scalar("SSIM_%d" % (t + 1), SSIM[t], epoch + 1)

        MSE_AVG  = sum(MSE)  / future_frames
        PSNR_AVG = sum(PSNR) / future_frames
        SSIM_AVG = sum(SSIM) / future_frames
        
        tensorboard_writer.add_scalar("MSE(val)",  MSE_AVG,  epoch + 1)
        tensorboard_writer.add_scalar("PSNR(val)", PSNR_AVG, epoch + 1)
        tensorboard_writer.add_scalar("SSIM(val)", SSIM_AVG, epoch + 1)

        print("Epoch {}, LOSS: {}, MSE: {} (x1e-3); PSNR: {}, SSIM: {}".format(
            epoch + 1, LOSS, 1e3 * MSE_AVG, PSNR_AVG, SSIM_AVG))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Moving-MNIST Conv-LSTM Training")

    ## Models
    parser.add_argument('--model', default = 'convlstm', type = str,
        help = 'The model is either \"convlstm\"" or \"convttlstm\".')
    parser.add_argument('--model-size', default = 'small', type = str,
        help = 'The model size is either \"small\" or \"large\".')

    parser.add_argument('--loss-function', default = 'l1', type = str, 
        help = 'The loss function for training.')

    ## Devices
    parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true',
        help = 'Use GPU for training.')
    parser.add_argument('--no-cuda', dest = 'use_cuda', action = 'store_false', 
        help = "Do not use GPU for training.")
    parser.set_defaults(use_cuda = True)

    parser.add_argument('--multi-gpu', dest = 'multi_gpu', action = 'store_true',
        help = 'Use multiple GPUs for training.')
    parser.add_argument('--single-gpu', dest = 'multi_gpu', action = 'store_false',
        help = 'Do not use multiple GPU for training.')
    parser.set_defaults(multi_gpu = True)

    ## Frame spliting
    parser.add_argument('--input-frames', default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future_frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')
    parser.add_argument('--output-frames', default = 19, type = int,
        help = 'The number of output frames of the model.')

    ## Folder paths
    parser.add_argument('--data-path', default = '../datasets/moving-mnist', type = str,
        help = 'The path to the folder storing the dataset.')
    parser.add_argument('--model-path', default = '../mnist/models', type = str,
        help = "The path to the folder storing the models.")
    parser.add_argument('--result-path', default = '../mnist/results', type = str,
        help = 'The path to the folder storing the results.')
    parser.add_argument('--model-name', default = "convlstm12", type = str,
        help = 'The model name is used to create the folder for models and results.')

    ## learning scheduling
    parser.add_argument('--num-epochs', default = 500, type = int, 
        help = 'Number of total epochs in training.')
    parser.add_argument('--start-epoch', default = 0, type = int, 
        help = 'Restart training from the specified epoch.')
    parser.add_argument('--save-epoch', default = 1, type = int, 
        help = 'Save the model parameters every save_epoch.')
    parser.add_argument('--batch-size', default = 4, type = int,
        help = 'The batch size in training.')

    # learning rate
    parser.add_argument('--learning-rate', default = 1e-3, type = float,
        help = 'Initial learning rate of the Adam optimizer.')
    parser.add_argument('--lr-decay-epoch', default = 5, type = int,
        help = 'Decay the learning rate every decay_epoch.')
    parser.add_argument('--lr-decay-rate', default = 0.99, type = float,
        help = 'Decay the learning rate by decay_rate every time.')

    # scheduled sampling ration
    parser.add_argument('--teacher-forcing', dest = 'teacher_forcing', action = 'store_true', 
        help = 'Use teacher forcing (scheduled sampling) in training phase.')
    parser.add_argument('--no-scheduled-sampling', dest = 'scheduled_sampling', action = 'store_false',
        help = 'Do not use teacher forcing (scheduled sampling) in training phase.')
    parser.set_defaults(teacher_forcing = True)

    parser.add_argument('--ssr-start-epoch', default = 100, type = int,
        help = 'Start scheduled sampling from ssr_start_epoch.')
    parser.add_argument('--ssr-decay-epoch', default = 1, type = int, 
        help = 'Decay the scheduled sampling every ssr_decay_epoch.')
    parser.add_argument('--ssr-decay-ratio', default = 4e-3, type = float,
        help = 'Decay the scheduled sampling by ssr_decay_ratio every time.')

    # logging
    parser.add_argument('--log-samples', default = 256, type = int,
        help = 'Log the statistics every specified number of samples.')
    parser.add_argument('--log-filename', default = "tensorboardX", type = str, 
        help = 'Log the training statistics in the specified file.')

    args = parser.parse_args()
    main(args)