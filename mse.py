import torch
import torch.nn.functional as F

tensor1 = torch.load('t1.pt',weights_only=True)

tensor2 = torch.load('t2.pt',weights_only=True)


# r1= torch.load('r1.pt',weights_only=True)
# r2= torch.load('r2.pt',weights_only=True)

# f1= torch.load('f1.pt',weights_only=True)
# f2= torch.load('f2.pt',weights_only=True)

# s1= torch.load('s1.pt',weights_only=True)
# s2= torch.load('s2.pt',weights_only=True)

# o1= torch.load('o1.pt',weights_only=True)
# o2= torch.load('o2.pt',weights_only=True)

# c1= torch.load('c1.pt',weights_only=True)
# c2= torch.load('c2.pt',weights_only=True)



print(f"Tensor1 shape: {tensor1.shape}")
print(f"Tensor2 shape: {tensor2.shape}")

if tensor1.shape == tensor2.shape:
    print("Shapes are equal.")
else:
    print("Shapes are different.")
    
mse= F.mse_loss(tensor1,tensor2)
print("MSE Loss ",mse.item())

# if r1.shape == r2.shape and f1.shape == f2.shape and s1.shape == s2.shape and o1.shape == o2.shape and c1.shape == c2.shape:
#     print("Shapes are equal.")
    
# else:
#     print("Shapes are different.")

# mse_r = F.mse_loss(r1, r2)
# mse_f = F.mse_loss(f1, f2)
# mse_s = F.mse_loss(s1, s2)
# mse_o = F.mse_loss(o1, o2)
# mse_c = F.mse_loss(c1, c2)

# print(f"MSE for r: {mse_r}")
# print(f"MSE for f: {mse_f}")
# print(f"MSE for s: {mse_s}")
# print(f"MSE for o: {mse_o}")
# print(f"MSE for c: {mse_c}")

