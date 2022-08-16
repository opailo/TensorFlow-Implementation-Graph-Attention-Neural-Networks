# Build the model
model = GraphAttentionModel(
    node_states, 
    edges, 
    HIDDEN_UNITS, 
    NUM_HEADS, 
    NUM_LAYERS, 
    OUTPUT_DIM)

# Compile the GAT
model.compile(loss = loss_fn,
              optimizer = optimizer,
              metrics = [accuracy_fn])



# Train the GAT
model.fit(x = train_indices,
      y = train_labels, 
      validation_split = VALIDATION_SPLIT,
      batch_size = BATCH_SIZE,
      epochs = NUM_EPOCHS,
      verbose = 2,
      )


# Test the model
_, test_accuracy = model.evaluate(x = test_indices,
                                  y = test_labels, 
                                  verbose = 1)

print("--" * 38 + f"\nTest Accuracy {test_accuracy*100:.1f}%")


# Prediction
test_probs = gat_model.predict(x=test_indices)

mapping = {v: k for (k, v) in class_index.items()}

for i, (probs, label) in enumerate(zip(test_probs[:10], test_labels[:10])):
    print(f"Example {i+1}: {mapping[label]}")
    for j, c in zip(probs, class_index.keys()):
        print(f"\tProbability of {c: <24} = {j*100:7.3f}%")
    print("---" * 20)