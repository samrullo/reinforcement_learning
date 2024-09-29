from towards_dqn.pong.pong_qlearning import PongQnet
import torch

model = PongQnet(6)

x = torch.randn(32, 4, 64, 64)
out = model(x)
print(f"out shape : {out.shape}")
