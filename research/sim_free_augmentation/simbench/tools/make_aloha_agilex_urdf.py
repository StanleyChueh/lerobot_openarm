"""Aloha-AgileX URDF for sfagen = RoboTwin 2.0 embodiment URDF + one fixed TCP frame per arm.
sfagen needs a URDF frame at the tool centre point; RoboTwin defines it in code, not in the URDF
(`envs/robot/robot.py::_trans_endpose`: TCP = ee link (`fl_link6` / `fr_link6`, config `ee_joints`) + gripper_bias
(0.12 m, `embodiments/aloha-agilex/config.yml`) along the link's x axis). Public robot data, not simulator truth.
Usage: make_aloha_agilex_urdf.py <embodiments/aloha-agilex dir>  -> writes urdf/arx5_sfagen.urdf next to the original."""
import sys, xml.etree.ElementTree as ET
from pathlib import Path
import yaml

d = Path(sys.argv[1]); cfg = yaml.safe_load((d / "config.yml").read_text())
src = d / cfg["urdf_path"]; tree = ET.parse(src); root = tree.getroot()
for side, joint in zip(("fl", "fr"), cfg["ee_joints"]):
    parent = next(j.find("child").get("link") for j in root.findall("joint") if j.get("name") == joint)
    ET.SubElement(root, "link", name=f"{side}_tcp")
    j = ET.SubElement(root, "joint", name=f"{side}_tcp_joint", type="fixed")
    ET.SubElement(j, "parent", link=parent); ET.SubElement(j, "child", link=f"{side}_tcp")
    ET.SubElement(j, "origin", xyz=f"{cfg['gripper_bias']} 0 0", rpy="0 0 0")
out = src.with_name("arx5_sfagen.urdf"); tree.write(out); print(out)
