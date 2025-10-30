from engine import Value
from nn import Neuron, Layer, MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

for k in range(200):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    n.zero_grad()
    loss.backward()

    for p in n.parameters():
        p.data += -0.1 * p.grad

    if k % 20 == 0:
        print(k, loss.data)

print("\nFinal Predictions:")
for x in xs:
    print(x, "=>", n(x).data)