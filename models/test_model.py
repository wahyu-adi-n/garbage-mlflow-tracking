import GarbageEffNetModelV0
import GarbageRes50NetModelV0
from torchinfo import summary

model_0 = GarbageEffNetModelV0()
summary(model_0)

model_1 = GarbageRes50NetModelV0()
summary(model_1)
