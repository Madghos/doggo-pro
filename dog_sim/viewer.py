from pathlib import Path
import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parent / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

# Ustaw pozycję początkową z keyframe "home"
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

if home_id == -1:
    raise RuntimeError("Nie znaleziono keyframe 'home'")

mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
