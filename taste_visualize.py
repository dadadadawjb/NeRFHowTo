import imageio
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='the experiment log path')
    parser.add_argument('--taste_type', type=str, help='the type of taste')
    parser.add_argument('--generate_type', type=str, help='the type of generated images')
    parser.add_argument('--gif', action='store_true', help='whether to output a gif')
    parser.add_argument('--mp4', action='store_true', help='whether to output a mp4')
    parser.add_argument('--fps', type=int, default=10, help='fps of the output video')
    args = parser.parse_args()

    rgb_frames, d_frames = [], []
    frame_names = sorted(os.listdir(os.path.join(args.path, 'taste', args.taste_type, args.generate_type)))
    for frame_name in frame_names:
        if frame_name.endswith("_rgb.png"):
            frame = imageio.imread(os.path.join(args.path, 'taste', args.taste_type, args.generate_type, frame_name))
            rgb_frames.append(frame)
        elif frame_name.endswith("_d.png"):
            frame = imageio.imread(os.path.join(args.path, 'taste', args.taste_type, args.generate_type, frame_name))
            d_frames.append(frame)
        else:
            continue
    
    if args.gif:
        imageio.mimsave(os.path.join(args.path, 'taste', args.taste_type, args.generate_type, 'rgb.gif'), rgb_frames, fps=args.fps)
        imageio.mimsave(os.path.join(args.path, 'taste', args.taste_type, args.generate_type, 'd.gif'), d_frames, fps=args.fps)
    if args.mp4:
        imageio.mimsave(os.path.join(args.path, 'taste', args.taste_type, args.generate_type, 'rgb.mp4'), rgb_frames, fps=args.fps)
        imageio.mimsave(os.path.join(args.path, 'taste', args.taste_type, args.generate_type, 'd.mp4'), d_frames, fps=args.fps)
