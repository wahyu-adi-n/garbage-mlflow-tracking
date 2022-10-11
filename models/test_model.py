from GarbageEffNetB0 import GarbageEffNetB0
from GarbageEffNetB7 import GarbageEffNetB7
from torchinfo import summary

effnet_b0 = GarbageEffNetB0()
summary(effnet_b0)

effnet_b7 = GarbageEffNetB7()
summary(effnet_b7)