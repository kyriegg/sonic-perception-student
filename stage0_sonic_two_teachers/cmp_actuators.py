import sys, types, json
if len(sys.argv) > 1 and sys.argv[1] == "instinct":  # Isaac Sim must start before torch/numpy are imported
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
sys.path.insert(0, "/mnt/d/robot paper/sonic_student" if sys.platform != "win32" else r"D:\robot paper\sonic_student")
from dump_actuators import table
from sonic_net import SONIC_JOINTS
which = sys.argv[1]
if which == "sonic":
    class Cfg:
        def __init__(self, **k): self.__dict__.update(k); self.__dict__.setdefault("armature", 0.0)
    fake = types.ModuleType("isaaclab"); 
    for name in ["isaaclab", "isaaclab.actuators", "isaaclab.assets", "isaaclab.assets.articulation", "isaaclab.sim", "isaaclab.utils"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["isaaclab.actuators"].ImplicitActuatorCfg = type("ImplicitActuatorCfg", (Cfg,), {})
    Cfg.InitialStateCfg = Cfg
    for n in ["ArticulationCfg"]: setattr(sys.modules["isaaclab.assets"], n, Cfg); setattr(sys.modules["isaaclab.assets.articulation"], n, Cfg)
    sim = sys.modules["isaaclab.sim"]
    for n in ["UrdfFileCfg", "RigidBodyPropertiesCfg", "ArticulationRootPropertiesCfg", "UsdFileCfg", "JointDrivePropertiesCfg", "UrdfConverterCfg"]: setattr(sim, n, Cfg)
    sim.UrdfConverterCfg.JointDriveCfg = Cfg; sim.UrdfConverterCfg.JointDriveCfg.PDGainsCfg = Cfg
    sys.modules["isaaclab.sim"].schemas = types.SimpleNamespace(RigidBodyPropertiesCfg=Cfg, ArticulationRootPropertiesCfg=Cfg)
    src = open("/home/robot/robot_paper/sonic/gear_sonic/envs/manager_env/robots/g1.py").read()
    g = {"__name__": "g1", "__file__": "/home/robot/robot_paper/sonic/gear_sonic/envs/manager_env/robots/g1.py"}
    try:
        exec(compile(src, "g1.py", "exec"), g)
    except Exception as e:
        print("exec stopped:", repr(e))
    acts = g["G1_CYLINDER_MODEL_12_DEX_CFG"].actuators
else:
    from instinctlab.assets import unitree_g1 as u
    acts = u.beyondmimic_g1_29dof_delayed_actuators
    print("instinct action scale == computed from non-delayed dict; check:")
    t2 = table(u.beyondmimic_g1_29dof_actuators, SONIC_JOINTS)
    from dump_actuators import resolve
    sc = resolve(u.beyondmimic_action_scale, SONIC_JOINTS)
    print({j: round(sc[j], 5) for j in SONIC_JOINTS} == {j: round(t2[j]["scale"], 5) for j in SONIC_JOINTS})
    json.dump(sc, open("D:/robot paper/sonic_student/act_instinct_used_scale.json", "w"), indent=1)
rows = table(acts, SONIC_JOINTS)
json.dump(rows, open(f"act_{which}.json".replace("act_", "/mnt/d/robot paper/sonic_student/act_" if sys.platform != "win32" else r"D:\robot paper\sonic_student\act_"), "w"), indent=1)
print("saved", len(rows))
