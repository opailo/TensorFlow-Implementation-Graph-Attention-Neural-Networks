class GraphAttentionModel(tf.keras.Model):
    def __init__(self, node_states, edges, hidden_units, num_heads, num_layers, output_dim, **kwargs,):

        super(). __init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation='relu')
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)


    def call(self, inputs): 
        node_states, edges = inputs
        x = self.preprocess(node_states)

        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x

        outputs = self.output_layer(x)
        return outputs
 
    
    def train_step(self, data):
        indices, labels = data

        # We're going to need to automatically keep track of the operations and be able to differentiate 
        # Use tf.GradientTape()
        with tf.GradientTape() as tape:
            
            # Forward
            outputs = self([self.node_states, self.edges])

            # Calculate the loss
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))

        # Calculate gradients
        grads = tape.gradient(loss, self.trainable_weights)

        # Update the weights
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update the metrics 
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        # return a dictionary with name and result pairs from the metrics
        return {m.name: m.result() for m in self.metrics}

    
    def predict_step(self, data):
        indices = data

        # Forward
        outputs = self([self.node_states, self.edges])

        # Return the softmax probabilities 
        return tf.nn.softmax(tf.gather(outputs, indices))


    def test_step(self, data):
        indices, labels = data

        # Forward
        outputs = self([self.node_states, self.edges])

        #  Calculate loss
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))

        # Update the metrics 
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        # return the same metrics as the training_step
        return {m.name: m.result() for m in self.metrics}
class GraphAttentionModel(tf.keras.Model):
    def __init__(self, node_states, edges, hidden_units, num_heads, num_layers, output_dim, **kwargs,):

        super(). __init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation='relu')
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)


    def call(self, inputs): 
        node_states, edges = inputs
        x = self.preprocess(node_states)

        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x

        outputs = self.output_layer(x)
        return outputs
 
    
    def train_step(self, data):
        indices, labels = data

        # We're going to need to automatically keep track of the operations and be able to differentiate 
        # Use tf.GradientTape()
        with tf.GradientTape() as tape:
            
            # Forward
            outputs = self([self.node_states, self.edges])

            # Calculate the loss
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))

        # Calculate gradients
        grads = tape.gradient(loss, self.trainable_weights)

        # Update the weights
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update the metrics 
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        # return a dictionary with name and result pairs from the metrics
        return {m.name: m.result() for m in self.metrics}

    
    def predict_step(self, data):
        indices = data

        # Forward
        outputs = self([self.node_states, self.edges])

        # Return the softmax probabilities 
        return tf.nn.softmax(tf.gather(outputs, indices))


    def test_step(self, data):
        indices, labels = data

        # Forward
        outputs = self([self.node_states, self.edges])

        #  Calculate loss
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))

        # Update the metrics 
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        # return the same metrics as the training_step
        return {m.name: m.result() for m in self.metrics}
