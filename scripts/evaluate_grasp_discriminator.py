from pathlib import Path

import numpy as np
import omegaconf
import torch
import trimesh



REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "GraspGenModels/checkpoints/graspgen_franka_panda.yml"
MESH_PATH = REPO_ROOT / "GraspGenModels/sample_data/meshes/mug.obj"

GRASPS = np.array(
    [
        [
            [1.0, 0.0, 0.0, 0.05],
            [0.0, 1.0, 0.0, 0.00],
            [0.0, 0.0, 1.0, 0.10],
            [0.0, 0.0, 0.0, 1.00],
        ],
        [
            [0.0, -1.0, 0.0, -0.05],
            [1.0, 0.0, 0.0, 0.00],
            [0.0, 0.0, 1.0, 0.10],
            [0.0, 0.0, 0.0, 1.00],
        ],
        [
            [-1.0, 0.0, 0.0, 0.00],
            [0.0, -1.0, 0.0, 0.05],
            [0.0, 0.0, 1.0, 0.10],
            [0.0, 0.0, 0.0, 1.00],
        ],
    ],
    dtype=np.float32,
)

cfg = omegaconf.OmegaConf.load(CONFIG_PATH)
MESH_SCALE = cfg.obj.scale
NUM_SAMPLE_POINTS = cfg.obj.num_sample_points
CHECKPOINT = CONFIG_PATH.parent / cfg.discriminator.checkpoint
if cfg.discriminator.checkpoint_object_encoder_pretrained is not None:
    cfg.discriminator.checkpoint_object_encoder_pretrained = str(
        CONFIG_PATH.parent / cfg.discriminator.checkpoint_object_encoder_pretrained
    )

if not MESH_PATH.exists():
    raise FileNotFoundError(f"Mesh file does not exist: {MESH_PATH}")
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Discriminator checkpoint does not exist: {CHECKPOINT}")

object_mesh = trimesh.load(MESH_PATH)
object_mesh.apply_scale(MESH_SCALE)
point_cloud, _ = trimesh.sample.sample_surface(object_mesh, NUM_SAMPLE_POINTS)
point_cloud = np.asarray(point_cloud, dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
point_cloud = torch.from_numpy(point_cloud).to(device).float()
point_cloud_center = point_cloud.mean(dim=0)
point_cloud = (point_cloud - point_cloud_center[None]).contiguous()

grasps = torch.from_numpy(GRASPS).to(device).float()
grasps[:, :3, 3] -= point_cloud_center
grasps[:, 3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
grasps = grasps.contiguous()

from grasp_gen.models.discriminator import GraspGenDiscriminator
model = GraspGenDiscriminator.from_config(cfg.discriminator).to(device)

if CHECKPOINT is not None and str(CHECKPOINT).lower() != "null":
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"Loaded discriminator checkpoint: {CHECKPOINT}")
else:
    print("No discriminator checkpoint supplied; using randomly initialized weights.")

model.eval()
data = {
    "points": point_cloud.unsqueeze(0).contiguous(),
    "grasps": grasps.unsqueeze(0).contiguous(),
}

with torch.inference_mode():
    outputs, _, _ = model.infer(data)

logits = outputs["logits"][0, :, 0].detach().cpu().numpy()
confidences = outputs["grasp_confidence"][0, :, 0].detach().cpu().numpy()

print(f"mesh: {MESH_PATH}")
print(f"num_sample_points: {NUM_SAMPLE_POINTS}")
for i, (logit, confidence) in enumerate(zip(logits, confidences)):
    print(f"grasp {i}: logit={logit:.6f}, confidence={confidence:.6f}")
