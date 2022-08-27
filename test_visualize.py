import imageio
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='the experiment log path')
    parser.add_argument('--gif', action='store_true', help='whether to output a gif')
    parser.add_argument('--mp4', action='store_true', help='whether to output a mp4')
    parser.add_argument('--fps', type=int, default=10, help='fps of the output video')
    args = parser.parse_args()

    frames = []
    frame_names = sorted(os.listdir(os.path.join(args.path, 'test')))
    for frame_name in frame_names:
        if frame_name.endswith("_show.png"):
            frame = imageio.imread(os.path.join(args.path, 'test', frame_name))
            frames.append(frame)
        else:
            continue
    
    if args.gif:
        imageio.mimsave(os.path.join(args.path, 'test', 'test_show.gif'), frames, fps=args.fps)
    if args.mp4:
        imageio.mimsave(os.path.join(args.path, 'test', 'test_show.mp4'), frames, fps=args.fps)
