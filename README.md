## Transformer Implementation from ([Vaswani et al.](https://arxiv.org/abs/1706.03762))

# What Are Graphs
<p align="left">
<img src="data/PIC 1.png"/>
</p>

Graphs are simply nodes connected to one another by edges. 
*  `G = (V, E)`

The information about the connections in a graph can be represented in an adjacency matrix
<p align="left">
<img src="data/PIC 2.png"/>
</p>

* Here connections between two nodes are depicted as `0` if there is NOT a connection and `1` if there IS a connection

The graph can also possess feature vectors for each individual node and edge.
* For example if the graph is a molecule:
  * The node could represent an atom where the features could be depicted as number of protons, neutrons, and electrons
  * Each each could represent a bond between atoms where the edge features are the type of bond

## What are Graphs Used For
* Pharma companies often represent the structure of their drugs in the form of graphs 
* Eccomerce companies may use large knowledge graphs to build reccommender systems for new products or services based on previous shopping habits
* Social media companies use graphs in which nodes are people and the edges represent connections to others. They can use these graphs to reccommend new individuals to follow based on node features such as common followers, interests, etc.
* Graph structures are also used in 3D graphics in which they are modeled as a polygon mesh made up of verteces, edges, and faces


# What Are Graph Neural Networks

Graphs can be used in machine learning applications in a number of ways
* First is simply node predictions
  * This involves predicting attributes or features about unlabeled nodes based on connections to other nodes
* Another is edge level predictions
  * This simply predicts if there will be a connection between two nodes in a graph
    * This is used by companies to predict which item a person is likely to purchase next
    * The company would then reccommend that item (Think Amazon reccommenders or Netflix reccomending new movies)
* Finally there is graph level predictions 
  * This is commonly done on drug data to classify if the molecule is a suitable drug or not 

Graph data is different from more common input data that may be given to neural networks
<p align="left">
<img src="data/PIC 3.png"/>
</p>

* Typical neural networks require the input data to be a set shape. However, graph data inputs will vary in shape by nature whether it be through total number of nodes or varrying connections between nodes. 
  * Image data can also be different in size and shape. But unlike images, graph data cannot be cropped or shrinked because it is size independent
    * When an image is sized down, data is lost and the matrix that makes up the image is changed
    * With graph data, you can't simply remove nodes to make it smaller as that would not be the same graph
* Another key characteristic of graph data is isomorphism
  * This means that a graph can look different, but still have the same structure
    * Imagine rotating a molecule 90-degrees. Although at first glance it may look like a different structure, it's in fact still the same structure just in a different perspective
    * The algo that takes in the graph data must be `permutation invarient`
      * This is actually the reason why the adjacency matrix can't be used as the input for the network because it is sensitive to changes in the node order
* Finally, graph data is `non-euclidian`
  * The euclidian distance between two nodes are not defined and so you can't tell how far Node-A is from Node-B
  
## The Goal of GNNs

* Representation learning
  * The idea is to learn a suitable representation of graph data 
    * Using all the information in the input graph (node features, adjacency matrix info on connections, etc.) the GNN will out put new representations called `node level embeddings` for each of the nodes 
      * These embeddings contain the structural and feature information on the other graph
      * This means each node in the output knows the information about the contents, context, and connections about all the other graphs in the representation
      * The nodes can then be used to form predictions 
* Predictions 
  * How these representaions are used depend heavily on the machine learning problem trying to be solved 
    * For example if you want to predict the label of a node, you can use that unlabeled node's embedding information on all the other labeled nodes and make a prediction based on the information
    * If you instead want to perform graph-level predictions, you would combine all the node embeddings in a certain way and get a representation of the graph
      * It's also common to run pooling operations on the representation graph to iteratively compress the graph into a representaitonal vector 

The size of node embeddings are going to be a changeable hyperparameter when creating the network
* For instance if the input graph have nodes with 50 features each, the output node embedding can have 128 features. 
  * Note that these features cannot be directly interpreted as they are the learned result of the model and a sort of compilation of all the various features about the input graph

# How do GNNs Work
<p align="left">
<img src="data/PIC 4.png"/>
</p>

Within the structure of GNNs, there are multiple `message passing layers`
* These layers are core building blocks of GNNs and are responsible for combine the node and edge information into the node embeddings

## Message Passing Layers 
These layers rely on a process known as `graph convolution`
* This process invovles taking in the input graph information such as node features, edge features, and connection information, combining it to get a new embedding, and updating the node features (states) with these embeddings
<p align="left">
<img src="data/PIC 5.png"/>
</p>

* For images, you are simply sliding learnable kernels over the matrix to extract the most import information from the image
  * This can be seen as combining the most pertinent information in a group of neighboring pixels into one feature for that particular area
* For non-euclidian graphs, this idea is extended by using the information in a particular node's neighborhood and combining it into one embedding
<p align="left">
<img src="data/PIC 6.png"/>
</p>

* Focusing on the yellow node in the image, assume we want to perform a graph convolution on the node
  * We would first pass the information from it's direct neighboring nodes 
    * This would give us information about the current node state `h1` and the neighboring node states `h2, h3, h4` from time step `k`
  * Next we would perform an aggregation on the neighboring node states 
  * Finally we would take the aggreagte and combine it witht he current node state to get an updated current node state `h1` in time step `k + 1`
* This process is repeated for each of the nodes in the graphh until there are updated embeddings for each node
<p align="left">
<img src="data/PIC 7.png"/>
</p>

* As you can see in the image, Node-5 only has green and blue information since it's only direct neighboring node is Node-4 which is blue. It doesn't yet know about the yellow Node-1
  * Another message passing step will change that since the update Node-4 will now have yellow information which it can then pass onto Node-5
* This can be repeated until every node has information about all the other nodes 
* The local aggregation of features can be compared to the learning of kernels in image convolutions

The number of message passing steps corresponds to the number of layers in the GNN (besides the input and output layers)

## Computation Graph Representation
It's possible to see how deep we are going into the graph from each node's perspective
<p align="left">
<img src="data/PIC 8.png"/>
</p>

* Already after two layers, the yellow node contains some info about all other nodes in the graph
* The number of layers in the GNN defines how many neighborhood hops a node will take
  * If a graph is small, it may only take a few layers to learn all the info about other nodes
* How ever if there are too many layers, `over-smoothing` will occur in which all nodes will be indistinguishable from each other

## Message Passing and Updating Functions

<p align="left">
<img src="data/PIC 9.png"/>
</p>

One particularly interesting version of Message Passing and Aggregation Functions is from a paper published by `Kipf and Welling` in which direct neighboring node info is aggregated as a sum of normalized neighbor embeddings. A self loop is then add that automatically incorporates the target current state node into the summation:
<p align="left">
<img src="data/PIC 10.png"/>
</p>

Another variant uses multi-layer perceptron (feed-forward neural networks) to create learnable weights
* This means the weights can be optimized so that the best possible aggregation is performed on the neighboring nodes 
<p align="left">
<img src="data/PIC 11.png"/>
</p>

Graph Attention Networks implement Attention Heads so that the model can consider which node features are most important when aggregating. As a result the final node embedding will contain the most important information about the other nodes
<p align="left">
<img src="data/PIC 12.png"/>
</p>

Gated Graph Neural Networks use reccurent units to update the states iteratively over time 
<p align="left">
<img src="data/PIC 13.png"/>
</p>

Most other variants of GNNs will only vary in the method of message passing and aggregation

## License

MIT License

Copyright (c) 2022 Otavio Pailo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


