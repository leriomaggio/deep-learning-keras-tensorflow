grads = K.gradients(loss, [w,b])
updates = [(w, w-lr*grads[0]), (b, b-lr*grads[1])]