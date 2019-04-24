import pysnooper
import torch


@pysnooper.snoop()
def number_to_bits(number):
    b = number + 3
    d = 123
    b = b + d
    return b


# number_to_bits(6)

a=torch.tensor(1)
b=torch.tensor(1.,requires_grad=True)
c=torch.tensor(1)
d=torch.tensor(1)

print(a)
print(b)
m=a+b
print(a)