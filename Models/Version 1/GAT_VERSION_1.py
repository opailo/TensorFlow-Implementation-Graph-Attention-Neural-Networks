#############################
# VERSION 1 NOT USED IN FINAL IMPLEMENTATION
#############################




class GAT(Layer):

  def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True, dropout=0.6, log_attention_weights=False):
    super().__init__()
    assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

    num_heads_per_layer = [1] + num_heads_per_layer # Allows to easily create GAT layers below

    gat_layers = [] # Collect the GAT layers

    for i in range(num_of_layers):
      layer = GATLayer(
          num_in_features = num_features_per_layer[i] * num_heads_per_layer[i], #consequence of concatentation
          num_out_features = num_features_per_layer[i + 1],
          num_of_heads = num_heads_per_layer[i + 1], 
          concat = True if i < num_of_layers - 1 else False, # Last GAT layer does mean avg while others perform concat
          activation = tf.keras.activations.elu() if i < num_of_layers - 1 else None, # Want last layer to output raw scores
          dropout_prob = dropout, 
          add_skip_connection = add_skip_connection, 
          bias = bias, 
          log_attention_weights = log_attention_weights
      )

      gat_layers.append(layer)

    self.gat_net == tf.keras.Sequential(
        *gat_layers,
    )

  # data is just a (in_nodes_features, edge_index) tuple
  def forward(self, data):
    return self.gat_net(data)


class GATLayer(Layer):
  # We'll use these constants in many functions so these will be placed here as member fields 
  src_nodes_dim = 0 # position of source nodes in edge index
  trg_nodes_dim = 1 # position of target nodes in edge index

  # These may change in the inductive the inductive setting (this isn't future proof)
  nodes_dim = 0 # node dimension (nodes dimension is the position of 'N' in tensor)
  head_dim = 1 # attention head dimension 

  def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=tf.keras.activations.elu(),
               dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

    super().__init__()

    self.num_of_heads = num_of_heads
    self.num_out_features = num_out_features
    self.concat = concat  # whether we should concatenate or average the attention heads
    self.add_skip_connection = add_skip_connection

    '''
    Trainable weights: linear projection matrics ("W" in paper)
    Attention target / source: "a" in paper 
    '''

    # Treat this one matrix as num_of_heads independent W matricies 
    self.linear_proj = tf.keras.layers.Dense(num_of_heads * num_out_features, input_shape=(num_in_features), use_bias=False, activation=None)
    
    '''
    Once you concat the target node (node i) and the source node (node j) we apply the 
    additive acoring function which will give the un-normalized score 'e'

    From there, split the 'a' vector 
      * So instead of doing [x, y] concatentation, followed by a dot-product with 'a',
      * We do a dot-product between x and 'a_left'
      * y and 'a_right' 
      * and then sum the two together 
    '''

    self.scoring_fn_target = tf.convert_to_tensor((torch.Tensor(1, num_of_heads, num_out_features)).numpy())
    self.scoring_fn_source = tf.convert_to_tensor((torch.Tensor(1, num_of_heads, num_out_features)).numpy())

    self.leakyReLU = tf.keras.layers.LeakyReLU(alpha = 0.3)

    self.dropout = tf.keras.layers.Dropout(rate = dropout_prob)

    self.log_attention_weights = log_attention_weights # Implemented later 
    self.attention_weights = None # Used for visualization later 

    self.init_params

  def forward(self, data):

    #  
    # STEP 1: Linear Projection + Regularization
    #

    in_nodes_features, edge_index = data # Unpack the data
    num_of_nodes = in_nodes_features.shape[self.nodes_dim]
    assert edge_index.shape[0] == 2, f'Expected edge index with shape = (2, E) got {edge_index.shape} instead'

    # shape - (N, FIN) 
      # N is the number of nodes in the graph
      # FIN is the number of input features per node 
    # We are going to apply dropout to all of the input node features (as in paper) but CORA features are already pretty sparse so I'm not sure how much it'll help
    in_nodes_features = self.dropout(in_nodes_features)

    # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT)
      # NH is the number of heads
      # FOUT is the number of output features 
    # We project the input node features into NH independent output features so that there is one for each attention head 
    nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

    nodes_features_proj = self.dropout(nodes_features_proj) # Dropout was performed in the paper here too

    #
    # STEP 2: Edge attention calculation
    #

    # Apply the scoring function (* represents element-wise multiplication)
    # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) since the summation operation squeezes out the last dimension 
    scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim = -1)
    scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim = -1)

    # shape = (E, NH, 1)
    attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
    # Stochastic is applied to the neighboring nodes aggregation operation
    attention_per_edges = self.dropout(attentions_per_edge)


    #
    # Step 3: Neighborhood aggregation
    #

    # Element-wise multiplication is performed 
    # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), the `1` gets broadcasted into FOUT
    nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

    # This part sums up the weighted and projected neighboring node feature vectors for every target node 
    # shape = (N, NH, FOUT)
    out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

    #
    # STEP 4: Residual / Skip connections, concatatenation and bias
    #

    out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
    return (out_nodes_features, edge_index)

  ##########################################
  # HELPER FUNCTIONS
  ##########################################


  def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
    """
    This function will perform softmax over the neighboring node features

    The attention scores for the current state nodes and their immediate connections are calculated as such:

        If we have a graph with six nodes and three nodes are connected to Node 4 (say nodes 1, 2, 3, 5) then we could 
        calculate the representation for Node 4 by taking into account the feature vectors of 1, 2, 3, and 5. 
        The function would calculate the attention scores using the already existing scores for each edge that connect directly to Node 4:

            2->4 / (1->4 + 2->4 + 3->4 + 5->4) 

        This is repreated for each unique connection to Node 4        
    """

    # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will boost the numerical stability)
    scores_per_edge = scores_per_edge - scores_per_edge.max()
    exp_scores_per_edge = scores_per_edge.exp() # softmax

    # Calculate the denominator shape = (E, NH)
    neighborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

    # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0)
    # due to the possibility of computer rounding a very small number all the way to 0
    attentions_per_edge = exp_scores_per_edge / (neighborhood_aware_denominator + 1e-16)

    # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with the projected node features 
    return attentions_per_edge.unsqueeze(-1)


  def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
    size = list(nodes_features_proj_lifted_weighted.shape) # convert to list otherwise assignment is not possible
    size[self.nodes_dim] = num_of_nodes # shape = (N, NH, FOUT)
    out_nodes_features = tf.zeros(size, dtype=in_nodes_features.dtype)

    return out_nodes_features


  def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
    """
    Lifts i.e. duplicates certain vectors depending on the edge index
    One of the tensor dims goes from N -> E (that's where the 'lift comes from)
    """

    src_nodes_index = edge_index[self.src_nodes_dim]
    trg_nodes_index = edge_index[self.trg_nodes_dim]

    # Using index_select is faster than 'normal' indexing (scores_source[src_nodes_index])
    scores_source = tf.gather_nd(scores_source, src_nodes_index, batch_dims = self.nodes_dim)
    scores_target = tf.gather_nd(scores_target, trg_nodes_index, batch_dims = self.nodes_dim)
    nodes_features_matrix_proj_lifted = tf.gather_nd(nodes_features_matrix_proj, src_nodes_index, batch_dims = self.nodes_dim)

    return scores_source, scores_target, nodes_features_matrix_proj_lifted


  def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
      # The shape must be the same as in exp_scores_per_edge: E -> (E, NH)
      trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

      # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
      size = list(exp_scores_per_edge.shape) # convert to list otherwise assignment isn't possible 
      size[self.nodes_dim] = num_of_nodes
      neighborhood_sums = tf.zeros(size, dtype=exp_scores_per_edge.dtype)

      # position i will contain a sum of exp scores of all the nodes that point to the node i 
      neighborhood_sums.scatter_add(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

      # Expand it again so that we can use it as a softmax denominator
        # node i's sum will be copied will be copied to all the locations where the source nodes pointed to i
        # shape = (N, NH) -> (E, NH)
      return tf.gather_nd(neighborhood_sums, trg_index, batch_dims = self.nodes_dim)


  def explicit_broadcast(self, this, other):
    # Append singleton dimensions until this.dim() == other.dim()
    for _ in range(this.dim(), other.dim()):
      this = tf.expand_dim(this, -1)

    #Explicitly exapnd so that shapes are the same
    return tf.broadcast_to(this, other)

  
  def init_params(self):

    tf.keras.initializers.GlorotUniform(self.linear_proj.weight)
    tf.keras.initializers.GlorotUniform(self.scoring_fn_target)
    tf.keras.initializers.GlorotUniform(self.scroring_fn_source)

    if self.bias is not None:
      torch.nn.init.zeros_(self.bias)


  def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
    if self.add_skip_connection: # add residual connection
      if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]: # if FIN == FOUT
          # unsqeeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
          # Basically we are copying input vectors NH times and adding to them to processed vectors 
          out_nodes_features += in_nodes_features.unqueeze(1)
      else:
          #FIN != FOUT then we need to project input feature vectors into a dimension that can be added to the output feature vectors
          # skip_proj adds lots of additional capacity which may lead in overfitting 
          out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

    if self.concat:
        # shape = (N, NH, FOUT) -> (N, NH*FOUT)
        out_nodes_features = out_nodes_features.view(-1, self.num_of_heads, self.num_out_features)
    else:
        # shape = (N, NH, FOUT) -> (N, FOUT)
        out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

    if self.concat:
        # shape = (N, NH, FOUT) -> (N, NH*FOUT)
        out_nodes_features = out_node_features.view(-1, self.num_of_heads * self.num_out_features)
    
    return out_nodes_features if self.activation is None else self.activation(out_nodes_features )