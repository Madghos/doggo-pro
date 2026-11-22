from pathlib import Path
import mujoco
import mujoco.viewer

SCENE = Path("dog_sim/scene.xml").resolve()

print("Loading:", SCENE)

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

print("\n=== MODEL INFO ===")
print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print("Keyframes:", model.nkey)

assert model.nu == 12, f"Expected 12 actuators, got {model.nu}"
assert model.nkey >= 1, "Missing home keyframe"

# Reset to home
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("\n=== JOINTS ===")
for i in range(model.nu):
    print(model.actuator(i).name, model.actuator_ctrlrange[i])

print("\nStarting MuJoCo viewer...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

print("Simulation closed.")
