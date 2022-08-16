def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data

def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph_data(training_config, device=None):
    dataset_name = training_config['dataset_name'].lower()
    should_visualize = training_config['should_visualize']

    if dataset_name == DatasetType.CORA.name.lower():

        node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
        node_labels_npy = pickle_read(os.path.join(CORA_PATH, 'node_labels.npy'))
        adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, 'adjacency_list.dict'))

        node_features_csr = normalize_features_sparse(node_features_csr)
        num_of_nodes = len(node_labels_npy)

        topology = build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True)


        if should_visualize: 
            plot_in_out_degree_distributions(topology, num_of_nodes, dataset_name)  
            visualize_graph(topology, node_labels_npy, dataset_name)

      
        topology = tf.convert_to_tensor(topology, dtype=tf.int64)
        node_labels = tf.convert_to_tensor(node_labels_npy, dtype=tf.int64)
        node_features = tf.convert_to_tensor(node_features_csr.todense())

        train_indices = tf.range(CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=tf.int64)
        val_indices = tf.range(CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=tf.int64)
        test_indices = tf.range(CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=tf.int64)

        return node_features, node_labels, topology, train_indices, val_indices, test_indices
    else:
        raise Exception(f'{dataset_name} not yet supported')


def normalize_features_sparse(node_features_sparse):
  assert sp.issparse(node_features_sparse), f'Expected a sparse matrix, got {node_features_sparse}.'

  node_features_sum = np.array(node_features_sparse.sum(-1)) 

  node_features_inv_sum = np.power(node_features_sum, -1).squeeze()

  node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1
  
  diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)

  return diagonal_inv_features_sum_matrix.dot(node_features_sparse)


def build_edge_index(adjacency_list_dict, num_nodes, add_self_edges=True):
  source_nodes_ids, target_nodes_ids = [], []
  seen_edges = set()
   
  for src_node, neighboring_nodes in adjacency_list_dict.items():
    for trg_node in neighboring_nodes:
      if (src_node, trg_node) not in seen_edges: 
        source_nodes_ids.append(src_node)
        target_nodes_ids.append(trg_node)

        seen_edges.add((src_node, trg_node))

  if add_self_edges:
    source_nodes_ids.extend(np.arange(num_nodes))
    target_nodes_ids.extend(np.arange(num_nodes))

  edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

  return edge_index