import torch

x = torch.arange(20, dtype=torch.float32)
print(x)

x = x.reshape(5,4)
print(x)

y = x.clone()
x = x + y
print(x)
print(x*y)
print(x.sum())
print(x.sum(axis=0))
print(x.sum(axis=1))
print(x.mean(axis=0))
print(x.mean(axis=1))

y = torch.ones(4)
print('matrix vector product is: ',torch.mv(x, y))
y = torch.clone(x)
print('matrix matrix product is: ',torch.mm(x, y.T))

u = torch.ones(3,4,5)
print('norm of x is:', torch.norm(u))

x = torch.arange(4, dtype=torch.float32, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
print('test grad:', x.grad)
y = x.sum()
x.grad.zero_()
y.backward()
print('test_grad:', x.grad)

