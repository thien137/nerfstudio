# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tools supporting the execution of COLMAP and preparation of COLMAP-based datasets for nerfstudio training.
"""
from pathlib import Path

from packaging.version import Version

from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

def run_megasam(
    image_dir: Path,
    megasam_dir: Path,
    megasam_path: Path,
    gpu: bool = True,
    verbose: bool = False,
    cvd: bool = True
) -> None:
    """Runs MEGASAM on the images.

    Args:
        image_dir (Path): _description_
        megasam_dir (Path): _description_
        camera_model (CameraModel): _description_
        gpu (bool, optional): _description_. Defaults to True.
        verbose (bool, optional): _description_. Defaults to False.
        cvd (bool, optional): _description_. Defaults to True.
        megasam_cmd (str, optional): _description_. Defaults to "megasam".
    """

    # Precompute mono-depth Depth-Anything
    mono_depth_cmd = [
        f"CUDA_VISIBLE_DEVICES=0 python {megasam_path}/Depth-Anything/run_videos.py",
        f"--encoder vitl",
        f"--load-from {megasam_path}/Depth-Anything/checkpoints/depth_anything_vitl14.pth",
        f"--img-path {image_dir}",
        f"--outdir {megasam_dir}/mono_depth",
    ]
    mono_depth_cmd = " ".join(mono_depth_cmd)
    with status(msg="[bold yellow]Running Megasam mono depth pre-processing...", spinner="runner", verbose=verbose):
        run_command(mono_depth_cmd, verbose=verbose)
        
    CONSOLE.log("[bold green]:tada: Done extracting Megasam mono depth.")
    
    # Precompute metric-depth Unidepth
    metric_depth_cmd = [
        f"CUDA_VISIBLE_DEVICES=0 python {megasam_path}/Unidepth/scripts/demo_mega-sam.py",
        #"--scene-name $seq",
        f"--img-path {image_dir}",
        f"--outdir {megasam_dir}/metric_depth",
    ]
    metric_depth_cmd = " ".join(metric_depth_cmd)
    with status(msg="[bold yellow]Running Megasam metric depth pre-processing...", spinner="runner", verbose=verbose):
        run_command(metric_depth_cmd, verbose=verbose)
    
    CONSOLE.log("[bold green]:tada: Done extracting Megasam metric depth.")
    
    # Running camera tracking
    camera_tracking_cmd = [
        f"CUDA_VISIBLE_DEVICE=0 python {megasam_path}/camera_tracking_scripts/test_demo.py",
        f"--datapath {image_dir}",
        f"--weights {megasam_path}/checkpoints/megasam_final.pth",
        #f"--scene_name $seq",
        f"--mono_depth_path {megasam_dir}/mono_depth",
        f"--metric_depth_path {megasam_dir}/metric_depth",
        f"--outdir {megasam_dir}",
    ]
    camera_tracking_cmd = " ".join(camera_tracking_cmd)
    with status(msg="[bold yellow]Running Megasam camera tracking...", spinner="runner", verbose=verbose):
        run_command(camera_tracking_cmd, verbose=verbose)

    CONSOLE.log("[bold green]:tada: Done Megasam camera tracking.")
    
    # Running consistent video depth optimization given estimated cameras
    
    # Run Raft Optical Flows 
    raft_cmd = [
        f"CUDA_VISIBLE_DEVICE=0 python {megasam_path}/cvd_opt/raft-things.py",
        f"--datapath {image_dir}",
        f"--model {megasam_path}/cvd_opt/raft-things.pth",
        f"--outdir {megasam_dir}/cache_flow"
        # f"--scene_name $seq",
        f"--mixed_precision",
    ]
    raft_cmd = " ".join(raft_cmd)
    with status(msg="[bold yellow]Running Megasam Raft Optical Flows...", spinner="runner", verbose=verbose):
        run_command(raft_cmd, verbose=verbose)
        
    CONSOLE.log("[bold green]:tada: Done Megasam Raft Optical Flows.")

    # Run CVD optmization
    cvd_opt_cmd = [
        f"CUDA_VISIBLE_DEVICE=0 python {megasam_path}/cvd_opt/cvd_opt.py",
        # f"--scene_name $seq",
        f"--output_dir {megasam_dir}/outputs_cvd",
        f"--w_grad 2.0 --w_normal 5.0",
    ]
    cvd_opt_cmd = " ".join(cvd_opt_cmd)
    with status(msg="[bold yellow]Running Megasam CVD optimization...", spinner="runner", verbose=verbose):
        run_command(cvd_opt_cmd, verbose=verbose)
        
    CONSOLE.log("[bold green]:tada: Done Megasam CVD optimization.")
    
# def megasam_to_json(
#     recon_dir: Path,
#     output_dir: Path,
#     camera_mask_path: Optional[Path] = None,
#     image_id_to_depth_path: Optional[Dict[int, Path]] = None,
#     image_rename_map: Optional[Dict[str, str]] = None,
#     ply_filename="sparse_pc.ply",
#     keep_original_world_coordinate: bool = False,
#     use_single_camera_mode: bool = True,
# ) -> int:
#     """Converts COLMAP's cameras.bin and images.bin to a JSON file.

#     Args:
#         recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
#         output_dir: Path to the output directory.
#         camera_model: Camera model used.
#         camera_mask_path: Path to the camera mask.
#         image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
#         image_rename_map: Use these image names instead of the names embedded in the COLMAP db
#         keep_original_world_coordinate: If True, no extra transform will be applied to world coordinate.
#                     Colmap optimized world often have y direction of the first camera pointing towards down direction,
#                     while nerfstudio world set z direction to be up direction for viewer.
#     Returns:
#         The number of registered images.
#     """

#     # TODO(1480) use pycolmap
#     # recon = pycolmap.Reconstruction(recon_dir)
#     # cam_id_to_camera = recon.cameras
#     # im_id_to_image = recon.images
#     cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
#     im_id_to_image = read_images_binary(recon_dir / "images.bin")
#     if set(cam_id_to_camera.keys()) != {1}:
#         CONSOLE.print(f"[bold yellow]Warning: More than one camera is found in {recon_dir}")
#         print(cam_id_to_camera)
#         use_single_camera_mode = False  # update bool: one camera per frame
#         out = {}  # out = {"camera_model": parse_colmap_camera_params(cam_id_to_camera[1])["camera_model"]}
#     else:  # one camera for all frames
#         out = parse_colmap_camera_params(cam_id_to_camera[1])

#     frames = []
#     for im_id, im_data in im_id_to_image.items():
#         # NB: COLMAP uses Eigen / scalar-first quaternions
#         # * https://colmap.github.io/format.html
#         # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
#         # the `rotation_matrix()` handles that format for us.

#         # TODO(1480) BEGIN use pycolmap API
#         # rotation = im_data.rotation_matrix()
#         rotation = qvec2rotmat(im_data.qvec)

#         translation = im_data.tvec.reshape(3, 1)
#         w2c = np.concatenate([rotation, translation], 1)
#         w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
#         c2w = np.linalg.inv(w2c)
#         # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
#         c2w[0:3, 1:3] *= -1
#         if not keep_original_world_coordinate:
#             c2w = c2w[np.array([0, 2, 1, 3]), :]
#             c2w[2, :] *= -1

#         name = im_data.name
#         if image_rename_map is not None:
#             name = image_rename_map[name]
#         name = Path(f"./images/{name}")

#         frame = {
#             "file_path": name.as_posix(),
#             "transform_matrix": c2w.tolist(),
#             "colmap_im_id": im_id,
#         }
#         if camera_mask_path is not None:
#             frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
#         if image_id_to_depth_path is not None:
#             depth_path = image_id_to_depth_path[im_id]
#             frame["depth_file_path"] = str(depth_path.relative_to(depth_path.parent.parent))

#         if not use_single_camera_mode:  # add the camera parameters for this frame
#             frame.update(parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id]))

#         frames.append(frame)

#     out["frames"] = frames

#     applied_transform = None
#     if not keep_original_world_coordinate:
#         applied_transform = np.eye(4)[:3, :]
#         applied_transform = applied_transform[np.array([0, 2, 1]), :]
#         applied_transform[2, :] *= -1
#         out["applied_transform"] = applied_transform.tolist()

#     # create ply from colmap
#     assert ply_filename.endswith(".ply"), f"ply_filename: {ply_filename} does not end with '.ply'"
#     create_ply_from_colmap(
#         ply_filename,
#         recon_dir,
#         output_dir,
#         torch.from_numpy(applied_transform).float() if applied_transform is not None else None,
#     )
#     out["ply_file_path"] = ply_filename

#     with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
#         json.dump(out, f, indent=4)

#     return len(frames)