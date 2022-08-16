class GraphAttention(layers.Layer):
    def __init__(self, units, 
                 kernel_initializer='glorot_uniform', # initializer used in paper
                 kernel_regularizer=None, **kwargs,):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)


    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape = (input_shape[0][-1], self.units),
            trainable = True, # gotta make sure we can actually train the layers and update the weights
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            name = 'kernel', 
        )

        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )

        self.built = True


    def call(self, inputs):
      node_states, edges = inputs

      # Linearly transform node states
      node_states_transformed = tf.matmul(node_states, self.kernel)

      # Computeattention scores
      node_states_expanded = tf.gather(node_states_transformed, edges)
      node_states_expanded = tf.reshape(node_states_expanded, (tf.shape(edges)[0], -1))

      # we're going to use leaky ReLU for calculating the attention scores
      # this will emphasize the most intense (i.e. important) values
      attention_scores = tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.kernel_attention))
      attention_scores = tf.squeeze(attention_scores, -1)
      
      # Normalize the attention scores 
      # Use clip_by_value() to force the tensors to be between -2 and 2
      attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
      attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,)
      attention_scores_sum = tf.repeat(
          attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], dtype='int32')))

      attention_scores_norm = attention_scores / attention_scores_sum


      # Gather the node states of neighboring nodes
      # Then apply attention scores and aggregate all of it together 
      node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
      out = tf.math.unsorted_segment_sum(
          data = node_states_neighbors * attention_scores_norm[:, tf.newaxis], 
          segment_ids = edges[:, 0],
          num_segments = tf.shape(node_states)[0])
      
      return out 
