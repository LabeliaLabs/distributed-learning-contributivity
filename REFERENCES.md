## Table of content

- [2CP: Decentralized Protocols to Transparently Evaluate Contributivity in Blockchain Federated Learning Environments](#2CP:-Decentralized-Protocols-to-Transparently-Evaluate-Contributivity-in-Blockchain-Federated-Learning-Environments)
- [CAPC Learning: Confidential and Private Collaborative Learning](#CAPC-Learning:-Confidential-and-Private-Collaborative-Learning)
- [Distributionally Robust Federated Averaging](#Distributionally-Robust-Federated-Averaging-(DRFA))
- [Towards Efficient Data Valuation Based on the Shapley Value](#Towards Efficient Data Valuation Based on the Shapley Value)
_ [Transparent Contribution Evaluation for Secure Federated Learning on Blockchain](#Transparent Contribution Evaluation for Secure Federated Learning on Blockchain)
- [Profit Allocation for Federated Learning](#Profit Allocation for Federated Learning)
- [Transparent Contribution Evaluation for Secure Federated Learning on Blockchain](#Transparent Contribution Evaluation for Secure Federated Learning on Blockchain)
- [Measure Contribution of Participants in Federated Learning](#Measure Contribution of Participants in Federated Learning)
- [FedCM: A Real-time Contribution Measurement Method for Participants in Federated Learning](#FedCM A Real-time Contribution Measurement Method for Participants in Federated Learning)
- [Incentive Mechanism for Reliable Federated Learning: A Joint Optimization Approach to Combining Reputation and Contract Theory](#Incentive Mechanism for Reliable Federated Learning A Joint Optimization Approach to Combining Reputation and Contract Theory)

---

## 2CP: Decentralized Protocols to Transparently Evaluate Contributivity in Blockchain Federated Learning Environments

- **Authors**: Harry Cai, Daniel Rueckert, Jonathan Passerat-Palmbach
- **Publication date:** Nov-15-2020

### Contribution:

- The paper proposes two new protocols for blockchain-based Federated Learning: **the Crowdsource Protocol** and **the Consortium Protocol**, both implemented using software framework **2CP**.
- The two protocols use **the step-by-step evaluation** strategy to quantify the relative *contributivity* of each partner, in which the individual updates are submitted to the blockchain and later 
- **The Crowdsource protocol** supposes the existence of an *Evaluator* and *Holdout Testset*.
- In **the Consortium protocol** such a **Holdout Testset** does not exist, so it adopts a new approach, labelled as **Parallel Cross Validation**.
  
### Experiments and results:

- The experiments were conducted using MNIST, with a training set of 60000 images split among 6 different trainers, that is 10000 images each.
  This is a bit problematic, because each trainer is expected to perform well with such a large portion of the training data. 
  This renders Contributivity measurements less informative.
- Both protocols performed well and gave similar results on *identical and uniformly distributed datasets* as well as datasets with varying sizes
unlike the Crowdsource protocol, the Consortium protocol struggles to give good results on datasets that had their labels flipped with various proportions
  
### Conclusion and future work:

- Evaluating the two protocols on datasets with unique distributions
- Designing a penalty scheme to protect against dishonest clients
- Studying the impact of **Differential Privacy** on contributivity scores

## CAPC Learning: Confidential and Private Collaborative Learning

- **Authors**:  Christopher A. Choquette-Choo, Natalie Dullerud, Adam Dziedzic, Yunxiang Zhang, Somesh Jha, 
Nicolas Papernot, Xiao Wang
- **Publication date:** Mar-19-2021

### Contribution:

- The paper proposes a collaborative and confidential learning protocol that improves on other techniques like **PATE** and **Federated Learning**.
- The protocol is *agnostic* to the data distribution and the machine learning models used by the participating parties.
- Learning is done through label sharing and not model weight aggregation.
- CaPC leverages *secure multiparty computation* **(MPC)**, *homomorphic encryption* **(HE)**, and other techniques in combination with privately aggregated teacher models to provide provable confidentiality and privacy guarantee.

### Experiments and results:

- CaPC improves the mean accuracy across both homogeneous and heterogeneous model architectures under the uniform and non-uniform data distribution setting
- The **privacy-utility trade-off** is determined by the number of parties involved in the protocol. Increasing the number of parties means we can issue more queries for a given privacy budget which leads to higher accuracy gains

## Distributionally Robust Federated Averaging (DRFA)

- **Authors**: Yuyang Deng, Mohammad Mahdi Kamani, Mehrdad Mahdavi
- **Publication date:** Feb-25-2021

### Contribution

- The paper proposes the **Distributionally Robust Federated Averaging** **(DRFA)** algorithm that is distributionally robust, while being communication-efficient via **periodic averaging**, and **partial node participation**.
- The main idea is to minimize *the empirical agnostic loss* to guarantee good performance over *the worst-combination of local distributions*.
- **DRFA** improves on the idea of empirical agnostic loss minimization which was first adopted by [the Agnostic Federated learning](https://arxiv.org/abs/1902.00146) algorithm.
  In the original approach, the server has to communicate with local clients at each iteration to update the global mixing parameter **λ**, which
  hinders its scalability due to communication cost. To cope with this issue, **DRFA** key technical contribution is a **randomized snapshotting schema**: **λ**, which controls
  the fraction of clients to participate in the next training round, is only updated periodically.   
- The gap between two consecutive **λ** is called the synchronizatin gap **τ**, which needs to be fine-tuned to guarantee fast convergence with minimal communication cost.
- **The agnostic federated learning algorithm (AFL)** can be considered a special case of DRFA when **the synchronization gap**(number of local updates in each training round) = 1.

### Experiments and results:

- **DRFA** achieves the same level of global accuracy as FedAvg while boosting the worst distribution accuracy
- **DRFA** outperforms **AFL**, **q-FedAvg** and **FedAvg** in terms of number of communications, and subsequently, wall-clock time required to achieve the same level of worst distribution accuracy (due to much lower number of communication needed) in a heterogeneous data setting.
- Pytorch implementation of the DRFA can be found [here](https://github.com/MLOPTPSU/FedTorch/).


## Towards Efficient Data Valuation Based on the Shapley Value
**Authors:** Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve Gurel, Bo Li, Ce Zhang, Dawn Song, Costas Spanos
2019
### Contribution:

### Experiments and results:
 
### Conclusion and future work:

## Profit Allocation for Federated Learning

**Authors:** Tianshu Song, Yongxin Tong, Shuyue Wei
2019
### Contribution:

### Experiments and results:
 
### Conclusion and future work:

## Transparent Contribution Evaluation for Secure Federated Learning on Blockchain

**Authors:** Shuaicheng Ma, Yang Cao, Li Xiong
2019
### Contribution:

### Experiments and results:
 
### Conclusion and future work:

## Measure Contribution of Participants in Federated Learning

**Authors** Guan Wang, Charlie Xiaoqian Dang, Ziye Zhou
2019

### Contribution:

### Experiments and results:
 
### Conclusion and future work:

## FedCM: A Real-time Contribution Measurement Method for Participants in Federated Learning
**Boyi Liu and Bingjie Yan, Yize Zhou, Jun Wang, Li Liu, Yuhan Zhang, Xiaolan Nie**
2021
### Contribution:

### Experiments and results:
 
### Conclusion and future work:

### Incentive Mechanism for Reliable Federated Learning: A Joint Optimization Approach to Combining Reputation and Contract Theory
**Jiawen Kang, Zehui Xiong, Dusit Niyato, Shengli Xie, Junshan Zhang**
2019

### Contribution:

### Experiments and results:
 
### Conclusion and future work:

