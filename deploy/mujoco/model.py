"""Convert the original training URDF primitives/inertias, reuse local Go2 visuals."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def numbers(values):
    return " ".join(str(float(v)) for v in values)


def origin(element):
    node = element.find("origin")
    if node is None:
        return {"pos": "0 0 0", "quat": "1 0 0 0"}
    roll, pitch, yaw = np.fromstring(node.get("rpy", "0 0 0"), sep=" ") / 2
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    quat = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
    return {"pos": node.get("xyz", "0 0 0"), "quat": numbers(quat)}


def load_model(config, root):
    source = Path(config["visual_source"]).expanduser()
    visual = ET.parse(source / "go2.xml").getroot()
    urdf = ET.parse(root / "resources/go2_description.urdf").getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    children = {}
    for joint in urdf.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    visual_bodies = {b.get("name"): b for b in visual.iter("body")}
    xml = ET.Element("mujoco", model="Go2 reference handstand")
    ET.SubElement(xml, "compiler", angle="radian", meshdir=str(source / "obj"),
                  autolimits="true", fusestatic="true")
    ET.SubElement(xml, "option", timestep=str(config["physics_dt"]),
                  gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic")
    defaults = ET.SubElement(xml, "default")
    ET.SubElement(defaults, "geom", friction=numbers([config["friction"], .005, .0001]),
                  condim="3")
    xml.append(deepcopy(visual.find("asset")))
    world = ET.SubElement(xml, "worldbody")
    ET.SubElement(world, "light", pos="0 -2 3", dir="0 1 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 .1",
                  rgba=".35 .38 .4 1")
    ET.SubElement(xml, "visual")
    motors = ET.SubElement(xml, "actuator")

    def add_link(name, parent, joint=None):
        body = ET.SubElement(parent, "body", name=name,
                             **(origin(joint) if joint is not None else {}))
        if joint is None:
            ET.SubElement(body, "freejoint", name="root")
        elif joint.get("type") == "revolute":
            limit = joint.find("limit")
            ET.SubElement(body, "joint", name=joint.get("name"), type="hinge",
                          axis=joint.find("axis").get("xyz"),
                          range=f'{limit.get("lower")} {limit.get("upper")}',
                          damping="0", armature="0", frictionloss="0")
            effort = float(limit.get("effort"))
            ET.SubElement(motors, "motor", name=joint.get("name"),
                          joint=joint.get("name"), ctrlrange=numbers([-effort, effort]))
        link = links[name]
        if name.endswith("_foot"):
            ET.SubElement(body, "site", name=name, size=".001", rgba="0 0 0 0")
        inertial = link.find("inertial")
        if inertial is not None and float(inertial.find("mass").get("value")) > 0:
            pose = origin(inertial)
            # All inertial origins in this source URDF have zero rotation.
            assert np.allclose(np.fromstring(pose.pop("quat"), sep=" "), [1, 0, 0, 0])
            inertia = inertial.find("inertia")
            ET.SubElement(body, "inertial", **pose,
                          mass=inertial.find("mass").get("value"),
                          fullinertia=" ".join(inertia.get(k) for k in
                                               ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]))
        old = visual_bodies.get("base_link" if name == "base" else name)
        if old is not None:
            for geom in old.findall("geom"):
                if geom.get("class") == "visual":
                    g = deepcopy(geom)
                    g.attrib.pop("class")
                    g.attrib.update(type="mesh", contype="0", conaffinity="0", group="2", density="0")
                    body.append(g)
        for index, collision in enumerate(link.findall("collision")):
            primitive = list(collision.find("geometry"))[0]
            kind = primitive.tag
            if kind == "box":
                size = np.fromstring(primitive.get("size"), sep=" ") / 2
            elif kind == "sphere":
                size = [float(primitive.get("radius"))]
            elif kind == "cylinder":
                kind = "capsule"  # Same replacement flag as the training URDF importer.
                size = [float(primitive.get("radius")), float(primitive.get("length"))/2]
            else:
                raise ValueError(f"Unsupported original collision: {name}/{kind}")
            ET.SubElement(body, "geom", name=f"{name}_collision_{index}", type=kind,
                          size=numbers(size), **origin(collision), group="3", density="0")
        for child_joint in children.get(name, []):
            add_link(child_joint.find("child").get("link"), body, child_joint)

    add_link("base", world)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    if not np.isclose(model.body_mass.sum(), config["robot_mass"], atol=1e-6):
        raise ValueError("MuJoCo mass differs from the training URDF")
    return model
