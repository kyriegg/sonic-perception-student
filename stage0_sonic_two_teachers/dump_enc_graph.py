import onnx, numpy as np
from onnx import numpy_helper
m = onnx.load("/home/robot/robot_paper/sonic/gear_sonic_deploy/policy/release/model_encoder.onnx")
init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
const = {}
for n in m.graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.t is not None and a.t.dims is not None:
                try: const[n.output[0]] = numpy_helper.to_array(a.t)
                except Exception: pass
def val(x):
    if x in init: v = init[x]
    elif x in const: v = const[x]
    else: return x
    return v.tolist() if v.size < 12 else f"<{v.shape}>"
for i, n in enumerate(m.graph.node):
    if n.op_type == "Constant": continue
    print(i, n.op_type, [val(x) for x in n.input], "->", list(n.output), [(a.name, onnx.helper.get_attribute_value(a)) for a in n.attribute if a.name in ("axis","perm","axes")])
    if n.op_type == "Gemm" and i > 140: break
