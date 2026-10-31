"""CPU inference for the exported actor, with no history or critic input."""
from pathlib import Path
import numpy as np


class Policy:
    def __init__(self, path, observation_dim=56, action_dim=12):
        self.observation_dim, self.action_dim = observation_dim, action_dim
        path = Path(path)
        self.onnx = path.suffix == ".onnx"
        if self.onnx:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(path), sess_options=options,
                                               providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        else:
            import torch
            torch.set_num_threads(1)
            self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self(np.zeros(self.observation_dim, dtype=np.float32))

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, self.observation_dim)
        if self.onnx:
            action = self.session.run(None, {self.input_name: obs})[0]
        else:
            import torch
            with torch.inference_mode():
                action = self.model(torch.from_numpy(obs)).numpy()
        if action.shape != (1, self.action_dim) or not np.isfinite(action).all():
            raise ValueError(f"Policy must return finite [1, {self.action_dim}] actions")
        return action[0]
