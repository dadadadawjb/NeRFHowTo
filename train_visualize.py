import imageio
import argparse
import os
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='the experiment log path')
    parser.add_argument('--gif', action='store_true', help='whether to output a gif')
    parser.add_argument('--mp4', action='store_true', help='whether to output a mp4')
    parser.add_argument('--fps', type=int, default=10, help='fps of the output video')
    parser.add_argument('--frames', type=int, default=-1, help='num of frames of the output video, -1 for all')
    args = parser.parse_args()

    frames = []
    frame_names = sorted(os.listdir(os.path.join(args.path, "train")))
    for frame_name in tqdm.tqdm(frame_names):
        if frame_name.endswith("_show.png"):
            frame = imageio.imread(os.path.join(args.path, "train", frame_name))
            frames.append(frame)
        else:
            continue
    
    if args.gif:
        if args.frames == -1:
            imageio.mimsave(os.path.join(args.path, "train", 'process_show.gif'), frames, fps=args.fps)
        else:
            imageio.mimsave(os.path.join(args.path, "train", 'process_show.gif'), frames[::len(frames)//args.frames], fps=args.fps)
    if args.mp4:
        if args.frames == -1:
            imageio.mimsave(os.path.join(args.path, "train", 'process_show.mp4'), frames, fps=args.fps)
        else:
            imageio.mimsave(os.path.join(args.path, "train", 'process_show.mp4'), frames[::len(frames)//args.frames], fps=args.fps)
