

def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
  """
    This function calculates only the degree distribution
    * It's possible to calculate many other metrics such as graph diameter, number of traingles, etc. using igraph / networkx
  """

  if isinstance(edge_index, tf.Tensor):
    edge_index = edge_index.cpu().numpy()

  assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}'

  # Store each node's input and output degree (they're the same for undirected graphs such as Cora)
  input_degrees = np.zeros(num_of_nodes, dtype=np.int)
  output_degrees = np.zeros(num_of_nodes, dtype=np.int)

  # Edge index shape = (2, E), the first row contains the source nodes, the second one contains the target nodes
  # source nodes  -> target nodes 
  num_of_edges = edge_index.shape[1]
  for cnt in range(num_of_edges):
    source_node_id = edge_index[0, cnt]
    target_node_id = edge_index[1, cnt]

    output_degrees[source_node_id] += 1 # source node points towards some other node -> increment its out degree
    input_degrees[target_node_id] += 1 # increment the input degrees

  hist = np.zeros(np.max(output_degrees) + 1)
  for out_degree in output_degrees:
    hist[out_degree] += 1

  fig = plt.figure(figsize=(12, 8), dpi=100) # otherwise plots are really small
  fig.subplots_adjust(hspace=0.6)

  plt.subplot(311)
  plt.plot(input_degrees, color='red')
  plt.xlabel('node id'); plt.ylabel('in-degree count'); plt.title('Input degree for different node ids')

  plt.subplot(312)
  plt.plot(output_degrees, color='green')
  plt.xlabel('node id'); plt.ylabel('out-degree count'); plt.title('Out degree for different node ids')

  plt.subplot(313)
  plt.plot(hist, color='blue')
  plt.xlabel('node degree')
  plt.ylabel('# nodes for a given out-degree') 
  plt.title(f'Node out-degree distribution for {dataset_name} dataset')
  plt.xticks(np.arange(0, len(hist), 5.0))

  plt.grid(True)
  plt.show()