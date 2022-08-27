# Datasets
> Determine: 
> 
> image array coordinate type: either 'right_down' or 'right_up', commonly 'right_down'
> 
> camera coordinate type: '{right/left}\_{up/down}\_{in/out}', related to the image when putting in front of the camera len
> 
> camera model: currently only support 'pinhole' type, with 'fx', 'fy', 'cx' and 'cy' intrinsic in pixels
> 
> camera pose type: either 'c2w' or 'w2c', as [R,t|0,1] extrinsic in meters with point's column vector convention


## helloworld
`tiny_nerf_data.npy` from <http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz>

|- `images`: np.ndarray((N, height, width, rgb)D), [0.0, 1.0], 'right_down'

|- `poses`: np.ndarray((N, 4, 4)D), c2w, 'right_up_out'

|- `focal`: float, pinhole camera model in pixels


## blender
`lego` from <http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip>

|- `{train/val/test}/*.png`: np.ndarray((height, width, rgba)D), [0, 255], 'right_down'

|- `transforms_{train/val/test}.json`: dict for camera

    |- `camera_angle_x`: float, $f_x = \frac{0.5 \times width}{\tan(0.5 \times camera\_angle\_x)}$, pinhole camera model in pixels

    |- `frames`: list(dict)

        |- `file_path`: corresponding image file path

        |- `rotation`: useless

        |- `transform_matrix`: np.ndarray((4, 4)D), c2w, 'right_up_out'

> maybe only valid in lego scene


> note: since original image is too large, you can use `resize_tool.py` in `utils` to resize the dataset, actually I resize to 0.5 of original size


## llff
`fern` from <http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip>

|- `images/*.JPG`: original high-resolution images, we do not use

|- `images_4/*.png`: down-sampled 4x images, we do not use

|- `images_8/*.png`: down-sampled 8x images, we use, np.ndarray((height, width, rgb)D), [0, 255], 'right_down'

|- `mpis4`: useless

|- `sparse`: useless

|- `database.db`: useless

|- `poses_bounds.npy`: need transformation, at last to np.ndarray((N, 4, 4)D), c2w, 'right_up_out', see `dataset.py` for details

|- `simplices.npy`: useless

|- `trimesh.png`: useless

> maybe only valid in fern scene


## colmap
`stuff` I take by myself

|- `images\*jpg`: np.ndarray((height, width, rgb)D), [0, 255], 'right_down'

|- `sparse`: sparse reconstruction results

    |- `0\*`: useless binary format

    |- `cameras.txt`: (camera_id, camera_model, width, height, fx, fy, cx, cy), pinhole camera model in pixels

    |- `images.txt`: (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name), w2c, 'right_down_in'

    |- `points3D.txt`: useless

> only valid when colmap setting the same with the following steps


You can use your own phone or other camera to capture real world images, then use [COLMAP](https://demuc.de/colmap/) to get their estimated poses as the ground truth to NeRF.

> tips: 
> 
> for modern phone, the resolution may be too much high, it is time-consuming, to turn down it, first you can change the image ratio to recommended 4:3 since modern phone start becoming long and thin, second you can turn off the AI mode to avoid AI refinement, third you can directly turn down the resolution, fourth you can use `resize_tool.py` in `utils` to resize the images before sending to colmap
> 
> for colmap, you can set your phone's camera fixed focals to use fixed intrinsic mode in colmap to get more estimated (image, pose) pairs


Recommended steps:
1. Take photos with fixed focal, height and width;
2. Create your dataset like `stuff` in `data` directory, put your images in `images` directory inside it;
3. Use `resize_tool.py` in `utils` to resize the images;
4. Use COLMAP to generate poses;
   ```bash
   colmap feature_extractor --database_path data/stuff/database.db --image_path data/stuff/images --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1
   colmap exhaustive_matcher --database_path data/stuff/database.db
   mkdir data/stuff/sparse
   colmap mapper --database_path data/stuff/database.db --image_path data/stuff/images --output_path data/stuff/sparse
   colmap model_converter --input_path data/stuff/sparse/0 --output_path data/stuff/sparse --output_type TXT
   ```


## custom
You can define your own dataset class by inheriting `Dataset` class in `dataset.py`, and implement your own `load_dataset()` and `observe_dataset()` functions, then use them in `train.py`, `test.py`, `taste.py` and `observe_dataset.py` correspondingly.
