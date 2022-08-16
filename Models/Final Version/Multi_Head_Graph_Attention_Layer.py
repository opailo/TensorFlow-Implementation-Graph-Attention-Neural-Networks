class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type='concat', **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    
    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [attention_layer([atom_features, pair_indices]) for attention_layer in self.attention_layers]
        # Concat or average the node states from each of the individual attention heads depending on the merge_type
        if self.merge_type == 'concat':
            outputs = tf.concat(outputs, axis = -1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis = -1), axis = -1)

        return tf.nn.relu(outputs)
        