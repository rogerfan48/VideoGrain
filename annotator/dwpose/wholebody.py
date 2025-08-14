import cv2
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose
from annotator.util import annotator_ckpts_path


class Wholebody:
    def __init__(self):
        device = 'cuda:0'
        providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']

        remote_dw_pose_path = "https://huggingface.co/sxela/dwpose_ckpts/resolve/main/dw-ll_ucoco_384.onnx"
        remote_yolox_path = "https://huggingface.co/sxela/dwpose_ckpts/resolve/main/yolox_l.onnx"
        
        dw_pose_path = os.path.join(annotator_ckpts_path, "dw-ll_ucoco_384.onnx")
        yolox_path = os.path.join(annotator_ckpts_path, "yolox_l.onnx")

        if not os.path.exists(dw_pose_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_dw_pose_path, model_dir=annotator_ckpts_path)
        if not os.path.exists(yolox_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_yolox_path, model_dir=annotator_ckpts_path)

        onnx_det = 'annotator/ckpts/yolox_l.onnx'
        onnx_pose = 'annotator/ckpts/dw-ll_ucoco_384.onnx'

        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
    
    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores