import configargparse

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    # general config
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--dataset_type', type=str, choices=['helloworld', 'blender', 'llff', 'colmap'], help='dataset type')
    parser.add_argument('--data_path', type=str, help='dataset path')
    parser.add_argument('--log_path', type=str, default="logs", help='log path')
    # model config
    parser.add_argument('--z_near', type=float, help='the accessible space nearest z value for all rays')
    parser.add_argument('--z_far', type=float, help='the accessible space farthest z value for all rays')
    parser.add_argument('--samples_num', type=int, help='num of samples along each ray for stratified sampling')
    parser.add_argument('--samples_num_prime', type=int, help='num of samples along each ray for hierarchical sampling')
    parser.add_argument('--x_freq_num', type=int, help='frequency num for x positional encoding')
    parser.add_argument('--d_freq_num', type=int, help='frequency num for d positional encoding')
    parser.add_argument('--x_freq_type', type=str, choices=["log", "linear"], help='frequency type for x positional encoding')
    parser.add_argument('--d_freq_type', type=str, choices=["log", "linear"], help='frequency type for d positional encoding')
    parser.add_argument('--width1', type=int, help='width1 of neural radiance field')
    parser.add_argument('--depth1', type=int, help='depth1 of neural radiance field')
    parser.add_argument('--width2', type=int, help='width2 of neural radiance field')
    parser.add_argument('--depth2', type=int, help='depth2 of neural radiance field')
    parser.add_argument('--width3', type=int, help='width3 of neural radiance field')
    parser.add_argument('--depth3', type=int, help='depth3 of neural radiance field')
    # train config
    parser.add_argument('--rays_num', type=int, help='num of rays per image during training, -1 for whole pixels rays')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--gamma', type=float, help='learning rate exponential decay')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--chunk_size', type=int, help='chunk size')
    parser.add_argument('--val_accelerate', action='store_true', help='whether to accelerate validation by using `rays_num`')
    parser.add_argument('--white_bkgd', action='store_true', help='whether to use white background')
    parser.add_argument('--verbose', action='store_true', help='whether to print details')

    args = parser.parse_args()
    return args
