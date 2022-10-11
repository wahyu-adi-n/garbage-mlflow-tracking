from GarbageEffNetModelV0 import GarbageEffNetModelV0
from GarbageConvTinyModelV0 import GarbageConvTinyModelV0
from torchinfo import summary

# model_0 = GarbageEffNetModelV0()
# summary(model_0)

model_1 = GarbageConvTinyModelV0()
summary(model_1)
