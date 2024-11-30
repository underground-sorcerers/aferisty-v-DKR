
# Fraud Detection with Graph Neural Networks (GNNs) and GraphSAGE

Where?
- banking
- e-commerce
- cybersecurity

Why?
 - Traditional ML focuses on individual transactions, but many cases of fraud involve more complex relationships between transactions, users, or entities. So, yeah, Graph Neural Networks (GNNs).

We are going to explore how to detect fraud in online banking using **GraphSAGE**, including:
- fitting a big dataset in memory
- preprocessing into graph format
- building + training + evaluating

---

## Motivation (Why even do graphs here?)

Fraud often occurs as in linked transactions or entities - that can provide the clues. For instance:
- more fraudulent transactions may share the same credit card;
- fraud often originates from the same IP-address;
- other interlinkings...

Graphs let us capture these relationships. By using GNNs (like GraphSAGE), we can aggregate information from neighboring nodes, learning embeddings that also represent relational structure.

---

## Dataset

The dataset (https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions/data) contains **unobfuscated** credit card transactions. See `preprocessing.py` and `fit_in_memory_experiments.ipynb` to know how we fit the part of the dataset in a small amount of memory to speed up the computations and bring out the most relevant parts and balances. The result is in `dataset_files/preprocessed_transactions.csv`.

---

## Preprocessing: Correct & Clean

### 1: Data Cleaning

We started by cleaning the data:
- Missing values are filled with the column means (though you might want to drop them out in some cases).
- Categorical columns (e.g. `Merchant Name`) are label-encoded into numerical values (however, depending on your situation, we could also suggest dropping those out or handling them in other way).
- Numerical features (e.g. `Amount`) are normalized.

```python
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
```

### 2: Balancing the Dataset

Fraudulent transactions are typically rare. To balance the dataset, we undersampled the genuine class (`Is Fraud?=0`) to match the size of the fraudulent class, but your possible alternative might be enforcing higher loss policy for missing the minority.

---

## Building the Graph

### 3: Define Nodes and Edges

Each transaction is a **node**. Edges are added between nodes that share specific features, such as the same `Card` or `User`. This way, the corresponding Graph ML task is **node classification**.

```python
# Edges are based on shared features
shared_features = ['User', 'Card']
for feature in shared_features:
    feature_dict = {}
    for index, value in df_balanced[feature].items():
        if value not in feature_dict:
            feature_dict[value] = []
        feature_dict[value].append(index)
    for nodes in feature_dict.values():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])
```

---

## Model: GraphSAGE

GraphSAGE learns node embeddings by sampling and aggregating information from neighbors (Well, as many GNNs do). This makes it scalable to large graphs: it doesn’t require full-batch operations.

### Architecture

Our model has two layers:
- the first layer aggregates information from 1-hop neighbors;
- the second layer aggregates information from 2-hop neighbors;
- we experimented with 3 and more, and it did not yield something useful specifically for our task;
- you might be interested in adding these if you have a larger dataset to fit.

```python
class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

## Training

We trained the model using the **negative log loss** function and the **Adam optimizer**. Here’s the (rather default) training loop:

```python
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## Evaluation

We expected to balance precision and recall (but secretly we favoured the recall):
- **Precision**: How many flagged transactions are actually fraudulent?
- **Recall**: How many fraudulent transactions are ever found and flagged?

```python
precision = precision_score(true_labels, pred, average='binary')
recall = recall_score(true_labels, pred, average='binary')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

### Results

- **Precision**: 0.8492
- **Recall**: 0.8470

### Confusion Matrix

As it happens very often, we visualize the confusion matrix:

```python
ConfusionMatrixDisplay.from_predictions(true_labels, pred)
plt.show()
```

---

## Conclusion

Using Graph Neural Networks like GraphSAGE, allows effectively leveraging relationships in transaction data to detect fraud. This method goes beyond traditional ML models by focusing on the connections between data points (that represent transactions and not users this time!).

---

### Maybe Later?

1. Attempt to run with larger graphs to catch more complex relationships (3+ hops).
2. Compare GraphSAGE with other GNN architectures like GAT or GCN.
3. Optimize hyperparameters, because you can optimize those forever :)
