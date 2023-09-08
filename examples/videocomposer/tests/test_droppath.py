from vc.models.droppath import DropPath
import mindspore as ms
from mindspore import ops
import numpy as np

def test():
    bs = 1
    d = 2
    #one_mask = ms.Tensor(np.ones([bs, 1]), ms.float32) 
    one_mask = ops.ones([bs, 1])
    
    x = ms.Tensor(np.ones([bs, d]))

    p_all_zero = p_all_keep = 0.2
    zero_mask = ops.Dropout(keep_prob=p_all_zero)(one_mask)[0] * p_all_zero
    keep_mask = ops.Dropout(keep_prob=p_all_keep)(one_mask)[0] * p_all_keep

    drop_prob = 0.5
    dp = DropPath(drop_prob)
    dp.set_train(True)

    ms.set_context(mode=0)
    
    print("zero mask: ", zero_mask)
    print("keep mask: ", keep_mask)
    for i in range(20):
        print(x+dp(x*2, zero_mask=zero_mask, keep_mask=keep_mask))

if __name__ == '__main__':
    test()
