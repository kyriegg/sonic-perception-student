"""Print per-joint stiffness/damping/armature/effort/action-scale of an actuator dict (no sim needed)."""
import re, sys, importlib, types, json, math

def resolve(spec, joints):
    out = {}
    for j in joints:
        for expr, v in (spec.items() if isinstance(spec, dict) else [(".*", spec)]):
            if re.fullmatch(expr, j):
                out[j] = v; break
    return out

def table(acts, joints):
    rows = {}
    for a in acts.values():
        names = [j for j in joints if any(re.fullmatch(e, j) for e in a.joint_names_expr)]
        for key in ("stiffness", "damping", "armature", "effort_limit_sim"):
            for j, v in resolve(getattr(a, key), names).items():
                rows.setdefault(j, {})[key] = v
        for j in names:
            rows[j]["type"] = type(a).__name__ + (f"(delay {a.min_delay}-{a.max_delay})" if hasattr(a, "max_delay") else "")
    for j in rows:
        r = rows[j]; r["scale"] = 0.25 * r["effort_limit_sim"] / r["stiffness"]
    return rows
