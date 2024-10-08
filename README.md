# Source code of TrustGuard Published in IEEE TDSC
Jie Wang, Zheng Yan, Jiahe Lan, Elisa Bertino, Witold Pedrycz, "[TrustGuard: GNN-based Robust and Explainable Trust Evaluation with Dynamicity Support](https://arxiv.org/pdf/2306.13339.pdf)," IEEE TDSC, 2024.


## Model architecture
<div align="center">
    <img src="./Model architecture.png" width="40%">
</div>

> TrustGuard is designed with a layered architecture that contains a snapshot input layer, a spatial aggregation layer, a temporal aggregation layer, and a prediction layer. In the snapshot input layer, a dynamic graph is segmented into a series of snapshots based on a time-driven strategy, and the snapshots are arranged in a chronological order for further analysis. The spatial aggregation layer focuses on the structural information (i.e., trust relationships) in each given snapshot to generate the spatial embedding of each node. It achieves this by aggregating information from the first-order and high-order neighbors of the node with the help of a defense mechanism, while also considering the dual roles of the node. Furthermore, the temporal aggregation layer is designed to capture temporal patterns from a sequence of snapshots. A position-aware attention mechanism is employed herein to learn attention scores, which indicate the importance of each timeslot to the target timeslot. By employing a weighted sum to the spatial embeddings calculated across all snapshots, we can obtain the final embedding of each node that contains both spatial and temporal features. In the prediction layer, a Multi-Layer Perception (MLP) is constructed to transform the final embeddings of any two nodes into a directed trust relationship.

## How to start
### Step 1: Configure python environment
```shell
torch==1.13.0
torch-geometric==2.2.0
torch-scatter==2.1.0
torch-sparse==0.6.16
```

### Step 2: Run the code
```shell
cd ./TrustGuard/code
python main.py
```

### Further information
* You can change the dataset, prediction tasks, and hyperparameters in arg_parser.py.
* We randomly initialize node embeddings for simplicity. You can use [Node2Vec](https://dl.acm.org/doi/pdf/10.1145/2939672.2939754) to improve prediction performance.
* "bitcoinotc.csv", sourced from [SNAP dataset](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html), is used for getting the snapshot's index at different timeslots. "bitcoinotc-rating.txt" is obtained by processing "bitcoinotc.csv". Specifically, in "bitcoinotc-rating.txt", the first two columns represent sourceID and targetID, respectively. The final two columns encode trust relationships using one-hot encoding: [1, 0] indicates a trusted relationship, and [0, 1] indicates a distrusted relationship. In addition, the data in "bitcoinotc-rating.txt" are in chronological order, with node indexes starting at 0. "bitcoinalpha-rating.txt" is obtained using the same setting.

## How to cite
If you find this work useful, please consider citing it as follows:
```bibtex
@ARTICLE{trustguard_tdsc,
  author={Wang, Jie and Yan, Zheng and Lan, Jiahe and Bertino, Elisa and Pedrycz, Witold},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={TrustGuard: GNN-Based Robust and Explainable Trust Evaluation With Dynamicity Support}, 
  year={2024},
  volume={21},
  number={5},
  pages={4433-4450},
  doi={10.1109/TDSC.2024.3353548}}
```

## Comments
If you have any questions about the code, please feel free to ask here or contact me via email at <jwang1997@stu.xidian.edu.cn>. This work is designed based on [Guardian](https://github.com/wanyu-lin/INFOCOM2020-Guardian), [DySAT](https://github.com/FeiGSSS/DySAT_pytorch), and [GNNGuard](https://github.com/mims-harvard/GNNGuard). Thanks for their excellent work!
