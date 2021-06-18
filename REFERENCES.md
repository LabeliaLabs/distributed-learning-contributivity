

## Table of content


- [2CP: Decentralized Protocols to Transparently Evaluate Contributivity in Blockchain Federated Learning Environments](#2CP:-Decentralized-Protocols-to-Transparently-Evaluate-Contributivity-in-Blockchain-Federated-Learning-Environments)
- [CAPC Learning: Confidential and Private Collaborative Learning](#CAPC-Learning:-Confidential-and-Private-Collaborative-Learning)
- [Distributionally Robust Federated Averaging](#Distributionally-Robust-Federated-Averaging-(DRFA))
---



## 2CP: Decentralized Protocols to Transparently Evaluate Contributivity in Blockchain Federated Learning Environments
### Contribution:
- The paper proposes two new protocols for blockchain-based Federated Learning: **the Crowdsource Protocol** and **the Consortium Protocol**, both implemented using software framework **2CP**.
- The two protocols use **the step-by-step evaluation** strategy to quantify the relative *contributivity* of each partner, in which the individual updates are submitted to the blockchain and later 
**The Crowdsource protocol** supposes the existence of an *Evaluator* and *Holdout testset*
In **the Consortium protocol** uses a new scheme, labelled as **Parallel Cross Validation**.
  
### Experiments and results:
- Both protocols performed well and gave similar results on *identical and uniformly distributed datasets* as well as datasets with varying sizes
unlike the Crowdsource protocol, the Consortium protocol struggles to give good results on datasets that had their labels flipped with various proportions 

### Conclusion and future work:
- Evaluating the two protocols on datasets with unique distributions
- Designing a penalty scheme to protect against dishonest clients
- Studying the impact of **Differential Privacy** on contributivity scores



## CAPC Learning: Confidential and Private Collaborative Learning
### Contribution:
- The paper proposes a collaborative and confidential learning protocol that improves on other techniques like **PATE** and **Federated Learning**.
- The protocol is *agnostic* to the data distribution and the machine learning models used by the participating parties.
- Learning is done through label sharing and not model weight aggregation.
- CaPC leverages *secure multiparty computation* **(MPC)**, *homomorphic encryption* **(HE)**, and other techniques in combination with privately aggregated teacher models to provide provable confidentiality and privacy guarantee.

### Experiments and results:
- CaPC improves the mean accuracy across both homogeneous and heterogeneous model architectures under the uniform and non-uniform data distribution setting
- The **privacy-utility trade-off** is determined by the number of parties involved in the protocol. Increasing the number of parties means we can issue more queries for a given privacy budget which leads to higher accuracy gains

## Distributionally Robust Federated Averaging (DRFA)

### Contribution 
- The paper proposes the **Distributionally Robust Federated Averaging** **(DRFA)** algorithm that is distributionally robust, while being communication-efficient via **periodic averaging**, and **partial node participation**.
- The main idea is to minimize *the empirical agnostic loss* to guarantee good performance over *the worst-combination of local distributions*.
- The global mixing parameter **λ** which is updated through a  **randomized snapshotting schema** controls the fraction of clients to participate in the next training round.
- **The agnostic federated learning algorithm (AFL)** can be considered a special case of DRFA when **the synchronization gap**(number of local updates in each training round) = 1.

### Experiments and results:
- **DRFA** achieves the same level of global accuracy as FedAvg while boosting the worst distribution accuracy
- **DRFA** outperforms **AFL**, **q-FedAvg** and **FedAvg** in terms of number of communications, and subsequently, wall-clock time required to achieve the same level of worst distribution accuracy (due to much lower number of communication needed) in a heterogeneous data setting.
- Pytorch implementation of the DRFA can be found [here](https://github.com/MLOPTPSU/FedTorch/).



