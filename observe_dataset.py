import os

from utils.config import config_parse
from utils.dataset import observe_helloworld, observe_blender, observe_llff, observe_colmap

if __name__ == '__main__':
    args = config_parse()
    print(args.expname)
    if not os.path.exists(os.path.join(args.log_path, args.expname)):
        os.mkdir(os.path.join(args.log_path, args.expname))
    else:
        pass
    if not os.path.exists(os.path.join(args.log_path, args.expname, 'dataset')):
        os.mkdir(os.path.join(args.log_path, args.expname, 'dataset'))
    else:
        print("dataset has already been observed")
        exit(-1)
    log_file = open(os.path.join(args.log_path, args.expname, 'dataset', 'log.txt'), 'w')

    # observe dataset
    print("start observing dataset")
    if args.dataset_type == 'helloworld':
        info = observe_helloworld(args.data_path, os.path.join(args.log_path, args.expname, 'dataset'))
        for item in info.items():
            print(item)
            print(item, file=log_file)
    elif args.dataset_type == 'blender':
        info = observe_blender(args.data_path, os.path.join(args.log_path, args.expname, 'dataset'))
        for item in info.items():
            print(item)
            print(item, file=log_file)
    elif args.dataset_type == 'llff':
        info = observe_llff(args.data_path, os.path.join(args.log_path, args.expname, 'dataset'))
        for item in info.items():
            print(item)
            print(item, file=log_file)
    elif args.dataset_type == 'colmap':
        info = observe_colmap(args.data_path, os.path.join(args.log_path, args.expname, 'dataset'))
        for item in info.items():
            print(item)
            print(item, file=log_file)
    else:
        raise NotImplementedError
    print("finish observing dataset")
