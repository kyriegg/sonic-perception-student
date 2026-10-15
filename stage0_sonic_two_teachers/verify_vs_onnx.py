"""Check sonic_net.py against the official deploy ONNX (encoder in g1 mode, decoder)."""
import sys, numpy as np, torch, onnxruntime as ort
sys.path.insert(0, "/mnt/d/robot paper/sonic_student")
from sonic_net import SonicActor, build_g1_encoder_input, quat_mul

root = "/home/robot/robot_paper/sonic/gear_sonic_deploy/policy/release/"
enc = ort.InferenceSession(root + "model_encoder.onnx")
dec = ort.InferenceSession(root + "model_decoder.onnx")
m = SonicActor.from_release("/mnt/d/robot paper/sonic_student/sonic_release_policy_sd.pt")
torch.manual_seed(0)
for trial in range(5):
    jp = torch.randn(1, 10, 29) * 0.5
    jv = torch.randn(1, 10, 29)
    rq = torch.nn.functional.normalize(torch.randn(1, 10, 4), dim=-1)
    pq = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
    x = build_g1_encoder_input(jp, jv, rq, pq)
    tok = m.encode_g1(x)
    # deploy layout: mode(4) | jp 10f (290) | jv 10f (290) | root_z 10f (10) | root_z (1) | anchor_ori (6) | anchor_ori 10f (60) | ...
    dif6 = x.view(1, 10, 64)[..., 58:]  # same 6D as training
    o = np.zeros((1, 1762), np.float32)
    o[0, 0] = 0.0  # encoder mode id (0 = g1)
    o[0, 4:294] = jp.flatten().numpy(); o[0, 294:584] = jv.flatten().numpy()
    o[0, 595:601] = dif6[0, 0].numpy(); o[0, 601:661] = dif6.flatten().numpy()
    tok_onnx = enc.run(None, {"obs_dict": o})[0]
    prop = torch.randn(1, 930)
    a = m.decode(tok, prop)
    a_onnx = dec.run(None, {"obs_dict": np.concatenate([tok.detach().numpy(), prop.numpy()], 1).astype(np.float32)})[0]
    print(f"trial {trial}: token max|diff| {np.abs(tok.detach().numpy() - tok_onnx).max():.2e} "
          f"(mismatching dims {(np.abs(tok.detach().numpy() - tok_onnx) > 1e-3).sum()}/64)  "
          f"action max|diff| {np.abs(a.detach().numpy() - a_onnx).max():.2e}")
