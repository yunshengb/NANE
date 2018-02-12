import tensorflow as tf

def masked_softmax_cross_entropy(preds, labels, mask=None, model=None):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    if mask is not None:
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
    return tf.reduce_mean(loss)



def print_mat(model, mat):
    result = mat[0:100,0:100]

    model.printer = tf.Print(result, [result],
                             message='result:\n',
                             summarize=100*100)