import os
from typing import Any, Tuple
try:
    import av
except ImportError:
    av = None
import torch
import numpy as np

try:
    from experimental.hwgat.pose_modules.Mediapipe import Model
except Exception:
    try:
        from pose_modules.Mediapipe import Model
    except Exception:
        Model = None

try:
    from experimental.hwgat.data_transform import (
        Compose,
        MediapipeDataProcess,
        PoseSelect,
        HandCorrection,
        NormalizeKeypoints,
        TemporalSample,
        WindowCreate,
    )
except ImportError:
    from data_transform import (  # type: ignore
        Compose,
        MediapipeDataProcess,
        PoseSelect,
        HandCorrection,
        NormalizeKeypoints,
        TemporalSample,
        WindowCreate,
    )

try:
    from ai_modules.hwgat.model_params import HWGATEParams
except ImportError:
    from experimental.hwgat.models.model_params import HWGATEParams  # type: ignore

class CFG():
    def __init__(self):
        cuda_id = 0
        self.mode = 'test'
        self.src_len = 192
        self.feature_type = 'keypoints'
        self.input_dim = 2

        self.model_type = 'HWGATE'

        self.device = torch.device(
            f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

        print("Running on device = ", self.device)

        self.model_params = HWGATEParams({'num_class' : 2002, 'src_len' : 192}, self.input_dim, self.device)

        self.batch_size = 1

        self.origin_idx = 0 #nose point
        self.anchor_points = [3, 4] #shoulder points
        self.left_slice = [9, 19, 7] #left hand slice
        self.right_slice = [19, 29, 8] #right hand slice

        kps = [0, 2, 5, 11, 12, 13, 14, 15, 16] + [0+33+468, 4+33+468, 5+33+468, 8+33+468, 9+33+468, 12+33+468, 13+33+468, 16+33+468, 17+33+468, 20+33+468,
             0+21+33+468, 4+21+33+468, 5+21+33+468, 8+21+33+468, 9+21+33+468, 12+21+33+468, 13+21+33+468, 16+21+33+468,
               17+21+33+468, 20+21+33+468]

        self.test_transform = self.val_transform = Compose([
            MediapipeDataProcess(),
            PoseSelect(kps, [0, 1]),
            HandCorrection(self.left_slice, self.right_slice),
            NormalizeKeypoints(self.origin_idx, self.anchor_points),
            TemporalSample(self.src_len),
            WindowCreate(self.src_len),
        ])
        
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        self.class_map_path = os.path.join(_pkg_dir, 'input', 'class_map_FDMSE.csv')
        if not os.path.exists(self.class_map_path):
            alt_map = os.path.abspath(os.path.join(_pkg_dir, '..', '..', 'models', 'class_map_hwgat.csv'))
            if os.path.exists(alt_map):
                self.class_map_path = alt_map

        ckpt_candidates = [
            os.environ.get("HWGAT_CHECKPOINT_PATH", "").strip(),
            os.path.join(_pkg_dir, 'input', 'model_best_loss.pt'),
            os.path.abspath(os.path.join(_pkg_dir, '..', '..', 'models', 'hwgat_best.pt')),
            os.path.abspath(os.path.join(os.getcwd(), 'models', 'hwgat_best.pt')),
        ]
        self.save_model_path = next((p for p in ckpt_candidates if p and os.path.exists(p)), os.path.join(_pkg_dir, 'input', 'model_best_loss.pt'))

def extract_frames(path):
    if av is None:
        raise ImportError("PyAV ('av' package) is required to decode video frames directly.")
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
    return frames

def get_video_features(vid_name: str) -> Tuple[Any, ...]:
    if not callable(Model):
        raise RuntimeError("pose_modules.Mediapipe Model is not available.")
    pose_model = Model()
    kp_shape = (543, 4)

    frames = extract_frames(vid_name)

    num_frames = len(frames)
    vid_height, vid_width = frames[0].shape[:2]

    features = np.zeros((num_frames, *kp_shape))
    i_th_frame = 0

    for frame in frames:
        # saving the i-th frame feature
        features[i_th_frame] = pose_model(frame)[0]
        i_th_frame += 1

    return vid_name, features, i_th_frame, vid_width, vid_height

def get_video_data(video):
    vid_name, features, num_frames, vid_width, vid_height = get_video_features(video)
    data = {
            'feat': features,
            'num_frames': num_frames,
            'vid_name': vid_name,
            'vid_width': vid_width,
            'vid_height': vid_height
            }
    return data
