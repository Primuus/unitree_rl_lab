"""Generate a separate folded-front-leg URDF without changing stock Go2 assets."""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

FRONT_POSE = {"hip": 0.0, "thigh": 1.75, "calf": -2.6}

def build(source: Path, destination: Path):
    root = ET.parse(source).getroot()
    expected = {f"{side}_{part}_joint" for side in ("FL", "FR") for part in FRONT_POSE}
    found = set()
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if name not in expected:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        if origin is None or origin.get("rpy", "0 0 0") != "0 0 0":
            raise ValueError(f"Unexpected source joint frame: {name}")
        part = name.split("_")[1]
        expected_axis = "1 0 0" if part == "hip" else "0 1 0"
        if axis is None or axis.get("xyz") != expected_axis:
            raise ValueError(f"Unexpected source joint axis: {name}")
        q = FRONT_POSE[part]
        origin.set("rpy", f"{q} 0 0" if part == "hip" else f"0 {q} 0")
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics", "mimic", "safety_controller"):
            for child in joint.findall(tag):
                joint.remove(child)
        found.add(name)
    if found != expected:
        raise ValueError(f"Missing front joints: {expected - found}")
    # Resolve meshes from the provided Unitree ROS installation. No copies or
    # edits to the original stock meshes, mass, inertia or rear-foot sphere.
    description_dir = source.parent.parent
    for element in root.iter():
        filename = element.get("filename")
        if filename and filename.startswith("package://go2_description/"):
            resolved = description_dir / filename.removeprefix("package://go2_description/")
            if not resolved.exists():
                raise FileNotFoundError(resolved)
            element.set("filename", str(resolved))
    movable = [j.get("name") for j in root.findall("joint") if j.get("type") in ("revolute", "continuous")]
    if len(movable) != 6 or any(not name.startswith(("RL_", "RR_")) for name in movable):
        raise ValueError(f"Expected exactly six hind-leg joints, got {movable}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    print(f"ASSET_OK: {len(found)} front joints fixed; {len(movable)} actuated; {destination}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("/workspace/unitree_ros/robots/go2_description/urdf/go2_description.urdf"))
    parser.add_argument("--dst", type=Path, default=Path(__file__).parent / "resources/go2_biped_walk.urdf")
    args = parser.parse_args()
    build(args.src, args.dst)
