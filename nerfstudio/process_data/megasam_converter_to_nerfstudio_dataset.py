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

"""Base class to processes a video or image sequence to a nerfstudio compatible dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from nerfstudio.process_data import megasam_utils, process_data_utils
from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import BaseConverterToNerfstudioDataset
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import install_checks
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class MegaSamConverterToNerfstudioDataset(BaseConverterToNerfstudioDataset):
    """Base class to process images or video into a nerfstudio dataset using colmap"""

    num_downscales: int = 1 # TODO
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""
    skip_image_processing: bool = True # TODO
    """If True, skips copying and downscaling of images and only runs COLMAP if possible and enabled"""
    megasam_path: Path = Path("colmap") # TODO
    """Path to the megasam executable"""
    cvd: bool = True
    """TODO"""
    gpu: bool = True
    """If True, use GPU."""
    include_depth_debug: bool = False
    """If --use-sfm-depth and this flag is True, also export debug images showing Sf overlaid upon input images."""
    same_dimensions: bool = True
    """Whether to assume all images are same dimensions and so to use fast downscaling with no autorotation."""
    use_single_camera_mode: bool = True
    """Whether to assume all images taken with the same camera characteristics, set to False for multiple cameras in colmap (only works with hloc sfm_tool).
    """

    @staticmethod
    def default_megasam_path() -> Path:
        return Path("megasam/")

    @property
    def absolute_megasam_path(self) -> Path:
        return self.output_dir / "megasam/"

    # def _save_transforms(
    #     self,
    #     num_frames: int,
    #     image_id_to_depth_path: Optional[Dict[int, Path]] = None,
    #     camera_mask_path: Optional[Path] = None,
    #     image_rename_map: Optional[Dict[str, str]] = None,
    # ) -> List[str]:
    #     """Save megasam transforms into the output folder

    #     Args:
    #         image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
    #         image_rename_map: Use these image names instead of the names embedded in the COLMAP db
    #     """
    #     summary_log = []
    #     if (self.absolute_megasam_path / "cameras.bin").exists():
    #         with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
    #             num_matched_frames = megasam_utils.megasam_to_json(
    #                 recon_dir=self.absolute_megasam_path,
    #                 output_dir=self.output_dir,
    #                 image_id_to_depth_path=image_id_to_depth_path,
    #                 camera_mask_path=camera_mask_path,
    #                 image_rename_map=image_rename_map,
    #                 use_single_camera_mode=self.use_single_camera_mode,
    #             )
    #             #summary_log.append(f"Megasam matched {num_matched_frames} images")
    #         #summary_log.append(megasam_utils.get_matching_summary(num_frames, num_matched_frames))

    #     else:
    #         CONSOLE.log("[bold yellow]Warning: Could not find existing MEGASAM results. Not generating transforms.json")
    #     return summary_log

    # def _export_depth(self) -> Tuple[Optional[Dict[int, Path]], List[str]]:
    #     """If SFM is used for creating depth image, this method will create the depth images from image in
    #     `self.image_dir`.

    #     Returns:
    #         Depth file paths indexed by COLMAP image id, logs
    #     """
    #     summary_log = []
    #     if self.use_sfm_depth:
    #         depth_dir = self.output_dir / "depth"
    #         depth_dir.mkdir(parents=True, exist_ok=True)
    #         image_id_to_depth_path = colmap_utils.create_sfm_depth(
    #             recon_dir=self.absolute_colmap_model_path
    #             if self.skip_colmap
    #             else self.output_dir / self.default_colmap_path(),
    #             output_dir=depth_dir,
    #             include_depth_debug=self.include_depth_debug,
    #             input_images_dir=self.image_dir,
    #             verbose=self.verbose,
    #         )
    #         summary_log.append(
    #             process_data_utils.downscale_images(
    #                 depth_dir,
    #                 self.num_downscales,
    #                 folder_name="depths",
    #                 nearest_neighbor=True,
    #                 verbose=self.verbose,
    #             )
    #         )
    #         return image_id_to_depth_path, summary_log
    #     return None, summary_log

    def _run_megasam(self):
        """
        Args:
            mask_path: Path to the camera mask. Defaults to None.
        """
        self.absolute_megasam_path.mkdir(parents=True, exist_ok=True)

        # set the image_dir if didn't copy
        if self.skip_image_processing:
            image_dir = self.data
        else:
            image_dir = self.image_dir

        # run megasam
        megasam_utils.run_megasam(
            image_dir=image_dir,
            megasam_dir=self.absolute_megasam_path,
            megasam_path=self.megasam_path,
            gpu=self.gpu,
            verbose=self.verbose,
            cvd=self.cvd     
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        # install_checks.check_ffmpeg_installed()
        # install_checks.check_colmap_installed(self.megasam_path) # TODO
