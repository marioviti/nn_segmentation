import numpy as np
from keras import backend as K
import tensorflow as tf
K.set_image_data_format('channels_last')


def weighted_categorical_crossentropy(target, output):
    return - tf.reduce_sum(target * tf.log(output), len(output.get_shape()) - 1)

def main():
    #test_target = tf.convert_to_tensor( np.array([[1.5,0.0]]), np.float32 )
    #test_output = tf.convert_to_tensor( np.array([[0.5,0.5]]), np.float32 )
    #test_target = tf.convert_to_tensor( np.array([[1.5,0.0],[0.0,1.5]]), np.float32 )
    #test_output = tf.convert_to_tensor( np.array([[0.5,0.5],[0.5,0.5]]), np.float32 )
    #sess = tf.InteractiveSession()  
    #loss = weighted_categorical_crossentropy(test_target,test_output)
    #print("\ntarget: {}\notuput: {}\nloss: {}".format(test_target.eval(),test_output.eval(),loss.eval()))
    #sess.close()
    
    test_target = tf.convert_to_tensor( np.array([[1.5,0.0],[0.0,1.5]]), np.float32 )
    test_output = tf.convert_to_tensor( np.array([[1.5,1.5],[0.5,0.5]]), np.float32 )
    sess = tf.InteractiveSession()
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=test_output,labels=test_target)
    print("\ntarget: {}\notuput: {}\nloss: {}".format(test_target.eval(),test_output.eval(),loss.eval()))
    sess.close()


if __name__ == "__main__":
    main()