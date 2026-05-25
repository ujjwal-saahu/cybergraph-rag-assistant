IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 7, JULY 2025

4847

## “Pre-Train, Prompt” Framework to Boost Graph Neural Networks Performance in EEG Analysis

Can-Ming Cui , Hong-Yi Chen , Man-Sheng Chen _, Graduate Student Member, IEEE_ , Jiahong Li, Zhaopeng Tong, Chanmei Fang, Chang-Dong Wang _, Senior Member, IEEE_ , and Yuexin Cai

_**Abstract**_ **—Electroencephalography (EEG) is a vital noninvasive technique used in neuroscience research and clinical diagnosis. However, EEG data have a complex nonEuclidean structure and are often scarce, making training effective graph neural network (GNN) models difficult. We propose a “pre-train, prompt” framework in graph neural networks for EEG analysis, called GNN-based EEG Prompt Learning (GEPL). The framework first uses unsupervised contrastive learning to pre-train on a large-scale EEG dataset. It then transfers the generic EEG knowledge learned by the model to target EEG datasets through graph prompt learning, thereby enhancing the model’s performance with a limited amount of EEG data from the target domain. We tested the framework on five EEG datasets, and the results showed that GEPL outperformed traditional fine-tuning methods in classification accuracy and area under the ROC curve (AUC). GEPL demonstrated improved generalization, robustness, and computational efficiency, thereby significantly reducing the overfitting risks associated with limited EEG data. Moreover, the model provided interpretable results, highlighting relevant brain regions during classification tasks. This research suggests that the “pre-train, prompt” paradigm is well-suited for EEG analysis and offers potential applications in other domains where data are limited.**

Received 29 October 2024; revised 20 January 2025; accepted 7 February 2025. Date of publication 13 February 2025; date of current version 4 July 2025. This work was supported in part by the National Key Research and Development Program of China under Grant 2023YFC2508400, in part by the National Natural Science Foundation of China under Grant 82271165 and Grant 62276277, in part by Open Project Program of Guangxi Key Laboratory of Digital Infrastructure under Grant GXDIOP2024011, in part by Guangzhou Science and Technology Project under Grant 2023A03J0711, and in part by Zhuhai Science and Technology Innovation Bureau in 2022 under Grant 222000400085. _(Corresponding authors: Chang-Dong Wang; Yuexin Cai.)_

Can-Ming Cui, Hong-Yi Chen, Man-Sheng Chen, and ChangDong Wang are with the School of Computer Science and Engineering, Sun Yat-Sen University, Guangzhou 510000, China, and with the Guangxi Key Laboratory of Digital Infrastructure, Guangxi Zhuang Autonomous Region Information Center, Nanning 530000, China, and also with the Guangdong Key Laboratory of Big Data Analysis and Processing, Guangzhou 510006, China (email: cuicm@mail2.sysu.edu.cn; chenhy359@mail2.sysu.edu.cn; chenmsh27@mail2.sysu.edu.cn; changdongwang@hotmail.com). Jiahong Li, Zhaopeng Tong, Chanmei Fang, and Yuexin Cai are with the Department of Otolaryngology Sun Yat-Sen Memorial Hospital, Sun Yat-Sen University, Guangzhou 510000, China (e-mail: ljh1995@hotmail.com; tongzhp@mail2.sysu.edu.cn; fangchm3@mail2.sysu.edu.cn; caiyx25@mail.sysu.edu.cn). Digital Object Identifier 10.1109/JBHI.2025.3541058

_**Index Terms**_ **—Electroencephalography, contrastive learning, graph neural networks, graph prompt learning, transfer learning.**

## I. INTRODUCTION

LECTROENCEPHALOGRAPHY (EEG) is a non- **E** invasive technique for recording brain electrical activity via electrodes on the scalp. It plays a crucial role in neuroscience and the clinical diagnosis of neurological disorders [1], [2], [3]. Analyzing EEG signals offers insights into brain mechanisms [4], cognitive processes [5], and aids in developing new treatments [6]. However, EEG data are inherently complex, with dynamic functional connectivity and a non-Euclidean structure, presenting challenges in analysis.

To address these challenges, machine learning methods, particularly graph neural networks (GNNs), have gained popularity in EEG research [7]. GNNs effectively capture the complex relationships between EEG channels, improving data analysis [8], [9]. However, significant challenges persist, most notably the scarcity of large, labeled EEG datasets. High acquisition costs, data privacy concerns, and inconsistent collection standards lead to data scarcity in EEG datasets [10]. This makes it difficult to train GNN models, limiting their performance in EEG analysis tasks.

Some studies use data augmentation to address EEG data scarcity [11], [12], [13], [14]. While this increases training data, it may introduce noise and fail to ensure data authenticity and diversity. Transfer learning provides an alternative by leveraging a small amount of target domain data along with prior knowledge from related domains to reduce data requirements [15], [16], [17]. For example, Nejedly et al. trained a general model on intracranial EEG (iEEG) data from one institution and fine-tuned it with data from another institution for artifact detection [18]. Wu et al. achieved cross-subject motor imagery EEG classification on small datasets using network initialization and fine-tuning strategies [19]. However, these methods, mainly based on convolutional neural networks (CNNs) or mathematical models, are limited by the type and format of EEG data, restricting cross-domain transfer learning and broader dataset use.

Furthermore, current transfer learning methods for EEG typically follow the “pre-train, fine-tune” paradigm. Several key issues arise when applying this paradigm to pre-trained GNN models [20]. Firstly, the inconsistency between the objectives of the pre-trained task and the downstream task may result in performance differences [21]. During the adaptation process of the downstream task, the pre-trained GNN model may suffer from the catastrophic forgetting problem [22], [23]. This

> 2168-2194 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 7, JULY 2025

4848

problem can become particularly severe in the EEG domain, where downstream datasets are generally small in size [24]. In such cases, the pre-trained model may overfit the downstream data, rendering the pre-training process ineffective.

Prompt learning [25] is an emerging approach in natural language processing (NLP) [26], which can effectively address the aforementionedchallenges.Itguidespre-trainedmodelstoadapt to new tasks by adjusting input prompts rather than fine-tuning model parameters. This approach retains the generalization capability of the pre-trained model, enabling efficient adaptation to new tasks. In recent years, the “pre-train, prompt” paradigm for GNNs has shown success in various applications [27], [28], [29].

In this paper, we propose adopting the “pre-train, prompt” transfer learning paradigm to EEG data. This approach aims to address the challenges faced by the “pre-train, fine-tune” paradigm to improve the performance of GNNs in target domains where EEG data are scarce. Designing graph prompts for EEG data can be challenging due to the need to consider variations in features across EEG datasets and to capture the influence of intricate functional connectivity in the brain during various tasks.

To address the challenges posed by the scarcity of EEG data and achieve efficient knowledge transfer, we introduce a “pretrain, prompt” framework in graph neural networks for EEG analysis, called GNN-based EEG Prompt Learning (GEPL). GEPL uses GNNs as the backbone and includes two main steps: (1) In the pre-training phase, we employ unsupervised contrastive learning [30] to pre-train the model on other EEG datasets, enabling it to acquire general knowledge from a large amount of unlabeled EEG data from other datasets. (2) In the prompt tuning phase, we adopt graph prompt learning to learn the target task using the pre-trained model and a small amount of labeled data from the target EEG dataset. These involve learnable graph prompts, including prompt features and prompt structure, optimized from both the feature space and functional connectivity perspectives. We conducted experiments on five EEG datasets, with target tasks including dementia detection [31], sudden deafness detection [32], and epilepsy detection [33]. The results show that compared with other fine-tuning methods, GEPL retains the generalization ability of the model and improves the fine-tuning efficiency through graph prompt learning. In addition, our framework also has good versatility and interpretability.

- Overall, the key contributions of this paper are as follows:

- r We introduced a “pre-train, prompt” framework in graph neural networks for EEG analysis, realizing efficient transfer learning and mitigating the impact of EEG data scarcity on GNNs. It employs contrastive learning and graph promptlearningtoachieveknowledgetransferacrossEEG domains, enhancing the performance of GNN models and tuning efficiency in downstream tasks.

- r Comparative experiments and visualization studies were conducted on different EEG datasets to demonstrate the superiority of our framework over traditional fine-tuning methods. Visualization of key parameters of the model reflects the interpretability and clinical significance of the framework. The model-agnostic nature of the framework makes it broadly applicable to other domains, driving potential advancements in medical research.

- r We collected resting-state EEG data from 425 patients with chronic tinnitus, 63 with sudden hearing loss, and 33 healthy controls at the Department of Otolaryngology, Sun Yat-sen Memorial Hospital. Two datasets were created: one for tinnitus patients and one for sudden hearing loss and controls, available at https://zenodo.org/records/ 13219018. These datasets provide valuable resources for studying the neural mechanisms of tinnitus and hearing loss, supporting future classification and prediction models.

## II. METHODOLOGY

## _A. Overview_

Fig. 1 provides an overview of our “pre-train, prompt” framework in graph neural networks for EEG analysis. We represent EEG as graph-structured data, termed EEG graphs. Nodes in the graph correspond to electrodes, and edges represent the functional connectivity between them. GNNs serve as the backbone of our proposed framework.

To enable GNNs to learn generic knowledge from EEG data across various domains, we employ unsupervised contrastive learning for pre-training the GNN model. We amalgamate multiple EEG datasets for pre-training, allowing the GNNs to obtain a more stable initialization.

We then employ graph prompt learning to improve the performance of the model in downstream tasks. During this phase, the pre-trained GNNs’ parameters are frozen to retain their generalization capability. Learnable graph prompts are used to modify the features and structure of EEG graphs in the target EEG dataset, forming prompt graphs. After encoding prompt graphs through GNNs, we obtain their representation _hG_ .

Finally, _hG_ is fed into a linear classifier _f_ classifier for downstream tasks. In the pre-training phase, all parameters of the GNN model are updated, while in the prompt tuning phase, only the parameters of the graph prompt and the linear classifier are updated. To summarize the above process, this methodology is termed GEPL. The notations that we will use throughout this paper are summarised in Table I.

## _B. Graph Neural Networks_

Graph neural networks are a type of neural network specifically designed to work with graph-structured data, capturing both the local and global relationships within the graph. GNNs iteratively update the feature representation of each node by aggregating features from its neighbors.

Let _h[ℓ] i_[denote the feature representation of node] _[ v][i]_[ at layer] _[ ℓ]_[.] The updated representation _h[ℓ] i_[+1] at the next layer is calculated using a transformation of the node’s own features _h[ℓ] i_[and][an] aggregation of features from its neighboring nodes _vj ∈ N_ ( _i_ ), where _N_ ( _i_ ) represents the set of neighboring nodes of _vi_ . The update rule is expressed as:

**==> picture [218 x 37] intentionally omitted <==**

where _WU[ℓ]_[and] _[ W][ ℓ] V_[are the learned weights of the GNN layer] _[ ℓ]_[,] _wij_ is a weight or attention score that determines the importance

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

CUI et al.: “PRE-TRAIN, PROMPT” FRAMEWORK TO BOOST GRAPH NEURAL NETWORKS PERFORMANCE IN EEG ANALYSIS

4849

**==> picture [360 x 239] intentionally omitted <==**

Fig. 1. Overview of workflow and GEPL. (a) EEG signals are resampled and cropped using a fixed-length window, then transformed using the fast Fourier transform to create a graph structure where electrodes serve as nodes and correlation coefficients form the adjacency matrix. In the pre-training dataset, the EEG signals from the subjects are segmented as extensively as possible, whereas in the downstream task dataset, each subject’s EEG signal is segmented into a single segment. (b) During the pre-training phase, contrastive learning is used to generate graph augmentations, while contrastive loss optimizes the model’s ability to generalize to EEG data. All parameters of the model are updated throughout this process. (c) In the target dataset, graph prompt tuning modifies node features and graph connections using learnable prompts to enhance task-specific performance. During this phase, only the parameters of the learnable graph prompt and the linear classification layer are updated, while the parameters of the pre-trained model remain unchanged.

TABLE I

NOTATIONS USED IN THIS PAPER TO DESCRIBE THE PROPOSED FRAMEWORK

**==> picture [199 x 231] intentionally omitted <==**

of node _vj_ ’s features to node _vi_ , and it can either be fixed or learned, and _σ_ is a non-linear activation function (e.g., ReLU or Sigmoid). This mechanism allows GNNs to effectively capture

the connectivity and feature distribution of the graph, making them suitable for a variety of tasks, such as node classification, link prediction, and graph classification.

The representations of the nodes in an EEG graph can be read out to obtain the graph-level representation _hG_ :

**==> picture [193 x 11] intentionally omitted <==**

where _{hi |vi ∈V}_ represents the set of feature vectors for all nodes _vi_ in the graph _G_ , and READOUT( _·_ ) is a permutationinvariant function such as summation, mean, or max pooling. This graph-level representation _hG_ is then used for subsequent tasks such as classification or regression.

## _C. Pre-Training Using Contrastive Learning_

We employ GraphCL [30], an unsupervised contrastive learning method, in the other EEG datasets to guarantee the effectiveness of training during the pre-training phase. In this phase, we utilize contrastive learning to take advantage of a large quantity of other EEG data, equipping the model with generic knowledge of EEG. Specifically, the contrastive learning in the pre-training phase contains four key components: (1) graph augmentation, _q_ ( _·_ ), applying transformations to the EEG graph to generate new augmented graphs, providing rich contrastive samples; (2) a _·_ graph encoder, _f_ ( ), for learning representations of EEG graphs. VariationsofGNNscanbeusedasgraphencodersinGEPL;(3)a projection function, constructed via a non-linear transformation _g_ ( _·_ ) to map the augmented representation to another space for calculating contrastive loss; (4) the corresponding contrastive loss function for optimizing the graph encoder.

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 7, JULY 2025

4850

Following the previous work [30], we generate two augmented graphs for each EEG graph _G_ using augmentations such as node dropout, edge dropout, and feature masking:

**==> picture [180 x 14] intentionally omitted <==**

Here, _G_[ˆ] _i_ and _G_[ˆ] _j_ are the augmented graphs of the original graph _G_ , generated by the augmentation functions _qi_ ( _·|G_ ) and _qj_ ( _·|G_ ), respectively.

The graph encoder _f_ ( _·_ ) encodes the augmented graphs _G_[ˆ] _i_ and _G_[ˆ] _j_ to obtain their representations, denoted _hGi_ and _hGj_ . After that, the projection function _g_ ( _·_ ) projects these graph representations into the contrastive learning space, generating the representations _zGi_ and _zGj_ .

In the pre-training phase, a small batch of _B_ graphs is randomly sampled, generating 2 _B_ augmented graphs through graph augmentation. In the contrastive learning space, the _b_ -th graph representation in the batch is represented by _zb,Gi_ and _zb,Gj_ , which are positive pairs. Negative pairs are not sampled, but generated from the remaining _B −_ 1 augmented graphs within the same batch. Given the positive and negative pairs, the InfoNCE contrastive loss [34] is applied to maximize the consistency of positive pairs and minimize the consistency of negative pairs:

**==> picture [241 x 57] intentionally omitted <==**

where _τ_ is a hyperparameter for controlling the range of the loss function. The final loss is calculated over all positive pairs in the mini-batch.

## _D. Prompt-Tuning in Target EEG Datasets_

We introduce graph prompt tuning and utilize a pre-trained model to perform prompt tuning on the target EEG datasets. Specifically, we integrate graph prompts with EEG graphs to create prompt graphs, and then obtain representations through the pre-trained model. Finally, we execute downstream tasks using a simple linear classifier. During this process, only the learnable graph prompt and the classifier are updated.

Prompt design in large language models involves inserting prompts or reorganizing the language within the model input [25]. These prompts modify the original semantics and structure of the input text, guiding the model to generate more accurate and relevant output. Similarly, we posit that prompts in graph models should function analogously by modifying the features and structure of the graph. Therefore, our graph prompt consists of two components: prompt features and prompt structure.

_1) Prompt Features:_ We employed a method similar to the Graph Prompt Feature (GPF) proposed by Fang et al. [35]. Learnable prompt vectors are added to the feature space of the nodes, and these prompt vectors are updated during the prompt tuning phase. The primary purposes of designing the prompt features are twofold: first, different EEG datasets may exhibit different distributions, and we aim to bridge the gap between different EEG datasets by altering the feature space of the nodes; second, we use frequency domain signals to construct node features, but the impact of different frequency bands of EEG

signals in various downstream tasks is not consistent. Therefore, we adjust the frequency domain signals through the prompt features to cater to the requirements of different downstream tasks.

Let _X_ represent the node feature matrix of an input EEG graph _G_ , which contains _n_ nodes. The features of node _vi_ are represented by a feature vector _xi_ . Therefore, the feature matrix is given by:

**==> picture [217 x 12] intentionally omitted <==**

where _F_ isthefeaturedimension.Thedimensionofthelearnable feature vector _p_ is also _F_ , which can be expressed as:

**==> picture [144 x 12] intentionally omitted <==**

In our approach, the prompt features consist of _k_ learnable prompt vectors, where _k_ is a hyperparameter. Typically, for more complex graph topologies (e.g., larger EEG graphs with more electrodes), a larger number of prompt vectors is needed to capture the variation in data distribution. The set of prompt vectors _P_ can then be represented as:

**==> picture [214 x 12] intentionally omitted <==**

For an input EEG graph, a specific prompt vector is generated for each node in the graph, and this generation process utilizes an attention mechanism. Specifically, through a learnable linear projection layer _a ∈_ R _[F][ ×][k]_ , the attention coefficient _α_ of _n_ nodes to _k_ prompt vectors is calculated:

**==> picture [174 x 29] intentionally omitted <==**

where _αi,j_ is the attention coefficient of node _vi_ to the prompt vector _pj_ . The prompt vector corresponding to each node is obtained by a weighted summation of _k_ prompt vectors based on attention coefficients. Specifically, for node _vi_ , the prompt vector is given by:

**==> picture [158 x 32] intentionally omitted <==**

Different from GPF [35], we multiply the prompt vector and the features of the nodes in the EEG graph item by item to get the feature matrix of a prompt graph instead of adding them together, because we found in experiments that this approach is more suitable for EEG signals in the frequency domain:

**==> picture [201 x 12] intentionally omitted <==**

where _◦_ denotes the Hadamard (element-wise) product.

_2) Prompt Structure:_ The EEG graphs constructed from the target datasets of downstream tasks are fully connected graphs, and the adjacency matrix _A_ represents the correlation coefficient matrix between the signals of electrodes/channels. However, not all connections in the graph are important for specific downstream tasks. To extract the connections crucial for these tasks and introduce sparsity into the graph, we use a learnable mask matrix _M_ to adjust the adjacency matrix _A_ , filtering out unimportant connections.

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

CUI et al.: “PRE-TRAIN, PROMPT” FRAMEWORK TO BOOST GRAPH NEURAL NETWORKS PERFORMANCE IN EEG ANALYSIS

4851

In particular, the adjacency matrix _A_ of an EEG graph _G_ with _n_ fully connected nodes is:

**==> picture [193 x 56] intentionally omitted <==**

TABLE II

STATISTICS OF DATASETS

**==> picture [220 x 62] intentionally omitted <==**

where _ai,j_ ⩾ 0 and _ai,j_ = 1 if _i_ = _j_ , as we are using the Pearson

The learnable mask matrix _M_ is:

**==> picture [200 x 56] intentionally omitted <==**

where _mij_ = 1 if _i_ = _j_ . Since the EEG graphs are undirected, both the adjacency matrices _A_ and the mask matrix _M_ are symmetric.

The final adjacency matrix _Ap_ , known as the prompt structure, is then calculated as follows:

**==> picture [170 x 11] intentionally omitted <==**

where _◦_ also denotes the Hadamard product, and the ReLU function ensures that there are no negative values in the final adjacency matrix.

The two components, the prompt features _Hp_ and the prompt structure _Ap_ , are then combined into one graph, referred to as the prompt graph _Gp_ . This prompt graph _Gp_ serves as the input to the pre-trained model during the prompt-tuning phase.

_3) Objective and Training:_ After obtaining the representation _hGp_ of the prompt graph _Gp_ through the pre-trained model _f_ pre-trained( _·_ ), we perform tuning by conducting a classification task with a linear classifier. During this phase, only the parameters of the prompt features _P_ , the prompt structure _M_ , and the classifier _f_ classifier( _·_ ) will be updated while the parameters of the pre-trained model remain fixed. The classifier is trained in a supervised learning setting, using cross-entropy loss as the objective function. This process is formulated as follows:

**==> picture [168 x 30] intentionally omitted <==**

where _y_ ˆ denotes the predicted probability of the classifier.

The cross-entropy loss function used for training the classifier

**==> picture [233 x 11] intentionally omitted <==**

where _y_ is the true label (ground truth).

## III. EXPERIMENTS AND RESULTS

## _A. Data and Preprocessing_

_1) Datasets:_ Our experiments were conducted on five realworld brain network datasets: TINNITUS, DEAP [36], SHL, ALZH [37], and TE [38]. The statistics of these datasets are shown in Table II.

The TINNITUS dataset is a proprietary dataset provided by Sun Yat-sen Memorial Hospital of Sun Yat-sen University,

comprising EEG recordings from 425 patients with chronic tinnitus. All EEG signals were collected using standard procedures. Participants were informed of the study’s purpose and ensured that they were in a relaxed state before the experiment. They sat in a soundproof and electromagnetically shielded room, keeping their eyes open and focusing on a computer screen, minimizing blinking, avoiding head movements, and staying awake. All electrode impedances were kept below 50 kΩ, and the EEG signals were sampled at a frequency of 128 Hz. The dataset indicates no statistical differences in demographic and other clinical information between the control and experimental groups. We filtered the signals to remove noise and redundant data, applied baseline correction, and re-referenced the data. Missing or damaged data were interpolated, and independent component analysis (ICA) was used to extract independent components. Finally, manual inspection ensured data purity. The TINNITUS dataset includes signals from 127 EEG channels, as shown in Fig. 2(a), with 2 reference electrodes placed on the mastoid.

DEAP is a multimodal physiological signal dataset used for emotion analysis, consisting of EEG and peripheral physiological signals from 32 participants while they watched 40 music videos. Each participant provided self-assessments of valence, arousal, dominance, and liking for each video. We used EEG recordings from all 32 participants. Each participant underwent 40 trials, resulting in a total of 1280 EEG recordings. The data werefirstdownsampledto128Hz,followedbytheapplicationof a 4.0–45.0 Hz bandpass filter. EOG artifacts were then removed, and the data were re-referenced to a common average. The EEG data included signals from 32 EEG channels, as shown in Fig. 2(b), and the EEG channels were reordered according to the Geneva sequence.

The SHL dataset, also from Sun Yat-sen Memorial Hospital, includes EEG recordings from 96 participants, comprising 63 sudden hearing loss patients and 33 healthy controls. EEG signals for each participant were recorded at a sampling rate of 128 Hz. The preprocessing procedures were consistent with those used for the TINNITUS dataset. The EEG recordings in this dataset include signals from 57 EEG channels, as shown in Fig. 2(c), with an average reference.

The ALZH dataset consists of resting-state EEG recordings with eyes closed from 88 subjects, including 36 Alzheimer’s patients, 23 frontotemporal dementia patients, and 29 healthy controls.Weused36Alzheimer’sdiseasepatientsand29healthy controlsforclassification.Cognitivefunctionwasassessedusing the Mini-Mental State Examination (MMSE). EEG recordings were collected using Nihon Kohden EEG 2100 equipment, with 19 scalp electrodes according to the international 10-20 system and 2 reference electrodes placed on the mastoids, at a sampling rate of 500 Hz. The distribution of the electrodes is shown in Fig. 2(d). Preprocessing steps included bandpass filtering

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 7, JULY 2025

4852

**==> picture [348 x 279] intentionally omitted <==**

Fig. 2. Electrode distribution of datasets.

(0.5–45 Hz), re-referencing, artifact correction using artifact subspace reconstruction (ASR), and ICA for artifact removal, transforming the 19 EEG channels into 19 ICA components.

The TE dataset comprises recordings from 71 healthy controls and 24 epilepsy patients, each lasting up to 8.5 minutes. The dataset uses the international 10-20 system, with a total of 19 electrodes and a sampling frequency of 500 Hz. The electrode distribution is shown in Fig. 2(e). Some electrodes use different reference electrodes to obtain different voltage values, resulting in a total of 35 EEG signals.

We combined the DEAP and TINNITUS datasets as the pretraining dataset and used the SHL, ALZH, and TE datasets as the target datasets for classification tasks.

_2) Data Preprocessing and EEG Graphs Constructing:_ First, EEG signals with different sampling rates are resampled to 200 Hz. Next, the EEG signals are cropped using a fixed-length window of 10,000 time points to ensure that all signals have the same sampling frequency and equal length. The fast Fourier transform (FFT) [39] is then applied to the EEG signals, preserving the logarithmic amplitude of the non-negative frequency components following previous studies [40].

In the pre-training dataset, EEG signals from each subject are segmented into multiple segments to increase data volume, enhancing the model’s generalization capabilities. For downstream task datasets, we extract a single segment per subject, simulating a data-scarce scenario that challenges the model to perform effectively with limited information, reflecting real-world situations where data may be scarce.

We construct each EEG signal as an EEG graph _G_ , where the nodes represent EEG electrodes/channels and the edges are determined by the correlations between the features of the nodes. The node features are the logarithmic amplitude of

the non-negative frequency components obtained from preprocessing. We calculate the Pearson correlation coefficients [41] between pairs of nodes and use these coefficients to form the adjacency matrix of the EEG graph. To introduce sparsity into the EEG graphs in the pre-training dataset, we retain only the edges corresponding to the top- _τ_ neighbors of each node and derive the adjacency matrix of the undirected graph through symmetrization. This process can be defined as follows:

**==> picture [203 x 25] intentionally omitted <==**

**==> picture [196 x 23] intentionally omitted <==**

where _X_ and _A_ represent the node feature matrix and the adjacency matrix of the EEG graph, respectively, the EEG graphs in the target datasets are fully connected. We modify the adjacency matrices using prompt structure to obtain sparse graphs.

## _B. Experiment Setting and Performance Validation_

We conducted experiments using three public datasets (DEAP [36], ALZH [37], TE [38]) and two private datasets (TINNITUS and SHL). We merged the larger datasets, DEAP and TINNITUS, and utilized this combined dataset for pretraining. Subsequently, the smaller datasets (SHL, ALZH, and TE) were used as target datasets for downstream classification tasks.

The experimental model consists of five layers of Graph Convolutional Networks [42], with the classification layer implemented as a single linear layer. During the pre-training phase, contrastive learning based on GraphCL [30] was employed

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

CUI et al.: “PRE-TRAIN, PROMPT” FRAMEWORK TO BOOST GRAPH NEURAL NETWORKS PERFORMANCE IN EEG ANALYSIS

4853

## TABLE III

PERFORMANCE COMPARISON ON THREE TARGET DATASETS

**==> picture [340 x 156] intentionally omitted <==**

using the combined dataset with a batch size of 10, a learning rate of 0.0001, and training for 100 epochs, during which the model with the minimum contrastive loss was saved. For the downstream tasks on the three target datasets, five-fold crossvalidation was used. In each fold, the data was divided such that 60% was allocated for training, 20% for validation, and 20% for testing. The model was trained for 500 epochs with a mini-batch size of 32, optimized using the Adam optimizer with an initial learning rate of 0.001, which decayed by a factor of ten every 100 epochs. A model checkpoint was saved whenever the validation loss decreased. Testing was performed on the test set, with performance metrics including classification accuracy and the area under the ROC curve (AUC). For performance validation, the results of each fold were averaged to obtain the overall results for a single experiment. Each experiment was repeated five times with different random seeds, and the average results were reported.

More details about the experiments and the source code can be found at the following GitHub link: https://github.com/cuicm/ GEPL.git.

## _C. Baselines_

To demonstrate the superiority of our GEPL, we chose common fine-tuning methods as baselines for the experiment and provide two classical classification models for reference, which helps us understand the impact of different fine-tuning strategies on the model’s performance on EEG datasets.

- 1) The **Decision Tree** [43] is a model that classifies EEG signals by learning decision rules based on their features.

- 2) The **Support Vector Machine (SVM)** [44] employs a Gaussian kernel function for the classification of EEG data.

- The fine-tuning methods for the baseline are as follows:

- 1) The **No Pre-training** method involves training and evaluating a randomly initialized backbone directly on the target EEG datasets. This demonstrates the performance of a model without pre-training. Most GNN-based EEG analysis methods adopt this approach [45], [46], [47].

- 2) The **Fine-tune** method involves fully parameterized finetuning of the pre-trained backbone on the target dataset,

 - leveraging pre-trained knowledge to improve the model’s performance. This approach is utilized in [48], [49].

- 3) The **MLP-1** method only updates the parameters of the single-layer linear classifier while keeping the pre-trained backbone unchanged. This aims to fully utilize the feature extraction capabilities of the pre-trained backbone, as seen in [50].

- 4) The **PARTIAL-** _l_ methods involve updating the parameters of the linear classifier and the last _l_ layers of the backbone, where _l_ = 1 _,_ 2 _,_ 3. This strategy progressively fine-tunes more layers, as utilized in [51], [52].

## _D. Performance Comparison and Statistical Analysis in Downstream Tasks_

We first report the mean ACC and mean AUC for the GEPL versus the baseline methods across three datasets, as shown in Table III. A paired _T_ -test was conducted on the three downstream datasets for statistical analysis. Our proposed GEPL consistently outperformed all other baseline methods in terms of both ACC and AUC.

For the SHL dataset, GEPL achieved a maximum ACC of 89.60 _±_ 2.65% at _k_ = 4 and a maximum AUC of 97.79 _±_ 2.69% at _k_ = 8. The accuracies of GEPL with _k_ = 4 are 35.20% ( _p <_ 0.001), 34.80% ( _p <_ 0.001), 33.40% ( _p <_ 0.001), 33.40% ( _p <_ 0.001), 30.60% ( _p <_ 0.001), and 19.40% ( _p <_ 0.005) higher than those of the PARTIAL-3, Fine-tune, PARTIAL-2, PARTIAL-1,MLP-1,andNoPre-trainingmethods,respectively. The improvements achieved by GEPL with _k_ = 8 in AUC over these baselines are 41.95% ( _p <_ 0.001), 46.04% ( _p <_ 0.001), 42.44% ( _p <_ 0.001), 45.08% ( _p <_ 0.001), 45.76% ( _p <_ 0.001), and 25.32% ( _p <_ 0.005) respectively.

For the ALZH dataset, GEPL achieved the highest ACC and AUC at _k_ = 1, 57.54 _±_ 3.96% and 62.48 _±_ 7.08%, respectively. The accuracies of GEPL with _k_ = 1 are 8.31% ( _p <_ 0.05), 8.00% ( _p <_ 0.05), 6.77% ( _p_ = 0.108), 4.62% ( _p <_ 0.05), 4.31% ( _p_ = 0.276), and 3.08% ( _p_ = 0.230) higher than those of the PARTIAL-2, PARTIAL-1, Fine-tune, No Pre-training, MLP-1, and PARTIAL-3 methods, respectively. The improvements in AUC over these baselines are 15.81% ( _p <_ 0.05), 13.58% ( _p <_ 0.05), 10.19% ( _p_ = 0.063), 2.48% ( _p_ = 0.72), 9.38% ( _p_ = 0.126), and 6.81% ( _p_ = 0.127), respectively.

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 7, JULY 2025

4854

**==> picture [407 x 115] intentionally omitted <==**

Fig. 3. Receiver operating characteristic (ROC) curves for three target datasets. The methods evaluated include Fine-Tune (FT), PARTIAL- _l_ (where _l_ ranges from 1 to 3), MLP-1, No Pre-training (No PT), and GEPL with different numbers of prompt vectors (from _k_ = 1 to _k_ = 8).

For the TE dataset, GEPL achieved the highest ACC of 80.42 _±_ 0.84% at _k_ = 2 and AUC of 75.26 _±_ 6.13% at _k_ = 1. The accuracies of GEPL with _k_ = 2 are 9.47% (p _<_ 0.05), 9.05% ( _p <_ 0.005), 7.37% ( _p <_ 0.05), 5.68% ( _p <_ 0.005), 2.53% ( _p_ = 0.117), and 2.31% ( _p <_ 0.05) higher than those of MLP-1, PARTIAL-3, PARTIAL-2, PARTIAL-1, Fine-tune, and No Pretraining methods, respectively. The improvements achieved by GEPL with _k_ = 1 in AUC over these baselines are 14.09% ( _p <_ 0.005), 14.26% ( _p <_ 0.05), 12.49% ( _p <_ 0.05), 14.66% ( _p <_ 0.05), 12.23% ( _p <_ 0.01), and 7.26% ( _p <_ 0.05) respectively.

These results indicate that GEPL enhances the performance of GNN models on EEG data with limited samples and is both robust and effective across different datasets. In addition, the average performance of the “pre-train, fine-tune” methods was even worse than the No Pre-training approach. This may be attributed to the limited number of samples in the target EEG datasets, which was not large enough to support effective fine-tuning. Consequently, this could have led to overfitting and other related issues. Such limitations not only break the prior knowledge embedded in the pre-trained backbone but also hinder effective knowledge transfer across different EEG datasets. In contrast, the “pre-train, prompt” framework is more suitable when EEG datasets are relatively scarce, as it mitigates the risk of overfitting and facilitates improved performance. Furthermore, GEPL kept relatively stable performance levels across different numbers of learnable prompt vectors ( _k_ ).

Fig. 3 presents the ROC curves for the GEPL and baseline methods across the SHL, ALZH, and TE datasets. For the SHL dataset (Fig. 3(a)), at a low false positive rate (FPR) of 10%, GEPL with _k_ = 8 achieves a true positive rate (TPR) of 91.5%, while the No Pre-training method reaches a TPR of 37.5%. The PARTIAL-2, PARTIAL-3, Fine-tune, PARTIAL-1, and MLP-1 methods exhibit TPRs of 18.8%, 18.0%, 16.5%, 14.6%, and 10.0%, respectively. For the ALZH dataset (Fig. 3(b)), at a low FPR of 40%, GEPL with _k_ = 1 achieves a TPR of 55.0%, whereas the No Pre-training, PARTIAL-3, and Fine-tune methods yield TPRs of 52.0%, 51.4%, and 45.7%, respectively. The MLP-1, PARTIAL-1, and PARTIAL-2 approaches produce TPRs of 44.5%, 41.7%, and 40.0%, respectively. For the TE dataset (Fig. 3(c)), at a low FPR of 20%, GEPL with _k_ = 8 secures a TPR of 43.5%, while the No Pre-training and Fine-tune methods achieve TPRs of 34.8% and 33.0%, respectively. The MLP-1, PARTIAL-3, PARTIAL-1, and PARTIAL-2 methods demonstrate TPRs of 26.0%, 22.0%, 21.7%, and 21.0%, respectively.

TABLE IV

COMPARISON OF THE NUMBER OF PARAMETERS ADJUSTED BY DIFFERENT TUNING STRATEGIES

**==> picture [233 x 134] intentionally omitted <==**

## _E. Analysis of Computational Complexity_

In this section, we analyze the computational complexity of our proposed GEPL framework against the conventional finetuning strategy. Table IV provides a clear comparison of how many parameters different tuning strategies have adjusted across the board over different datasets. Fine-tuning, which involves updating all the parameters in the model, serves as the baseline for our experiments. In contrast, GEPL only updates a subset of parameters consisting of the linear classifier, the prompt structure, and the _k_ learnable prompt vectors in the prompt features.

As shown in Table IV, GEPL significantly decreases the number of parameters to be adjusted compared to the fine-tuning approach. For example, when _k_ = 1, the number of parameters updated in the SHL dataset accounts for merely 0.11% of all parameters used in the fine-tuning strategy. As the value of _k_ increases, so is the increase in the number of parameters adjusted; however, as a fraction of the total, it is still very small. Specifically, when _k_ = 8, the number of parameters adjusted in the SHL dataset is still less than 1% of the total parameters needed for fine-tuning.

This substantial reduction in the number of parameters to be tuned translates to lower computational complexity and potentially faster training times. More importantly, this mirrors the model efficiency of GEPL in efficiently exploiting knowledge transferred from a previously trained model while reducing the needfor extensiveadjustments totheparameters. This maymake

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

CUI et al.: “PRE-TRAIN, PROMPT” FRAMEWORK TO BOOST GRAPH NEURAL NETWORKS PERFORMANCE IN EEG ANALYSIS

4855

TABLE V

RESULTS OF THE ABLATION STUDY ON THREE TARGET DATASETS

**==> picture [309 x 71] intentionally omitted <==**

TABLE VI

PERFORMANCE OF OUR FRAMEWORK WITH OTHER BACKBONE ENCODERS

**==> picture [280 x 60] intentionally omitted <==**

a big difference when dealing with EEG data, since the latter has limited samples and is prone to overfitting.

## _F. Ablation Study_

In this section, we present the results of an ablation study designed to evaluate the impact of different components of the proposed GEPL on classification performance. We conducted experiments by systematically removing specific components of the graph prompt and the datasets used for pre-training. The variants considered in this study are: (1) wo-PF (without prompt features), (2) wo-PS (without prompt structure), (3) wo-TINNITUS (without the TINNITUS dataset), (4) wo-DEAP (withouttheDEAPdataset),and(5)GEPL(fullframework).The results of these experiments are summarized in Table V.

Removing the prompt features (wo-PF) results in a significant decline in performance across all datasets. This indicates that the prompt features are essential for enhancing the model’s ability to effectively distinguish EEG categories. Similarly, the removal of the prompt structure (wo-PS) also leads to a decrease in performance, although the impact is not as severe as that of removing the prompt features. One possible reason for this is that the downstream tasks we performed are graph-level tasks, where node features are aggregated through average pooling to form the graph representation. This aggregation reduces the prominence of the prompt structure’s effect to some extent, as its impact is more pronounced in node-level tasks. However, the prompt structure still plays a vital role by introducing sparsity into the EEG graph, thereby reducing computational complexity. Excluding TINNITUS or DEAP from pre-training slightly degrades performance, indicating that good performance can still be achieved with a single pre-training dataset. Ablation studies demonstrate that each component of the GEPL model contributes to its overall performance. In conclusion, the prompt features are particularly important, while the prompt structure and diverse pre-training datasets provide additional, albeit different, levels of performance enhancement.

## _G. Performance With Other Backbone Encoders_

In this section, we use different graph neural networks as the backbone encoders for GEPL and analyze their impact on the

framework’s performance. The backbones considered include the Graph Convolutional Network (GCN) [42], the Graph Attention Network (GAT) [53], the Graph Isomorphism Network (GIN) [54], and GraphSAGE [55]. The results are summarized in Table VI.

It is evident that GEPL can achieve strong performance with limited EEG data by utilizing various backbone encoders, which highlights the versatility of the model-agnostic GEPL. In practical applications, researchers can select the most appropriate backbone encoder based on specific tasks. This flexibility in selection allows GEPL to identify the optimal configuration across a range of application scenarios.

## _H. Clinical Interpretability_

To investigate the interpretability of the proposed method in clinical medicine, we conducted a saliency visualization analysis on the best-performing model for each target dataset. Specifically, we performed classification tasks on all participants and plotted heatmaps that reflect the average saliency of the model for each node. Saliency maps [56] intuitively indicate which parts of the data the model considers most important. This not only helps evaluate whether our method accurately captures the key information from EEG data in different medical tasks but also provides valuable insights for practical medical applications. Fig. 4 presents the average saliency maps for all participants.

Fig. 4(a) shows the saliency map generated by the model during the classification task distinguishing between patients with sudden hearing loss and healthy subjects. The saliency map observed strong activation in the right frontal and parietal regions. This observation aligns with findings from other studies [57], [58], which indicate that sudden hearing loss impacts thefrontalregion.Furthermore,uniformactivationwasobserved in the left temporal lobe, suggesting functional changes in the auditory cortex related to auditory information processing [59]. These results further demonstrate that the model effectively learns and executes classification tasks based on brain regions associated with sudden hearing loss.

Fig. 4(b) shows the model’s attention to different brain regions when differentiating between patients with Alzheimer’s disease and healthy subjects. The model demonstrates a

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 7, JULY 2025

4856

**==> picture [464 x 145] intentionally omitted <==**

Fig. 4. Saliency maps of all participants for three target datasets. These saliency maps are for: (a) SHL; (b) ALZH and (c) TE.

particular focus on EEG activity in the parietal, occipital, and frontal regions (P3, O1, O2, Fp1, and Fp2). This observation aligns with the findings of Jacobs et al. [60], Thientunyakit et al. [61], and Johnson et al. [62], suggesting that the parietal, occipital, and frontal lobes are associated with Alzheimer’s disease. Damage or dysfunction in these areas can result in specific cognitive and behavioral impairments. The saliency map indicates that the model effectively concentrates on critical brain regions related to memory, attention, and executive functions in the Alzheimer’s disease classification task, thereby validating the soundness of our approach in Alzheimer’s disease diagnosis.

Fig. 4(c) displays the model’s attention to different brain regionswhendistinguishingbetweenepilepsypatientsandhealthy subjects. The model places significant emphasis on the prefrontal, lateral temporal, and posterior temporal areas (Fp1, T7, T8, P7, and P8), which are closely associated with epileptic EEG activity. The pronounced focus on the prefrontal region may be linked to prevalent prefrontal abnormalities observed in epilepsy patients [63], [64]. In these patients, certain neurons and neural circuits in the prefrontal cortex are prone to hyperexcitation, leading to abnormal synchronous discharges that can trigger seizures. Additionally, the model’s attention to the lateral temporal and posterior temporal regions is consistent with existing medical research on temporal lobe epilepsy [65], [66].

## _I. Visualization and Analysis of Feature Distributions_

We adopted a feature visualization method for deep neural networks (DNNs) called manifold discovery and analysis (MDA) [67] to visualize the distribution of data in the feature space as our model performs different downstream tasks. For each downstream dataset, we selected the model that demonstrated the best performance during cross-validation and visualized the distribution of data from the test set within the model’s feature space.

First, we conducted a visualization analysis of the five-layer GCN model, focusing on the feature distribution in layers 2 through 5. Fig. 5 presents the feature distributions of these four intermediate layers across three downstream datasets. It is evident that the model exhibits excellent classification capability in the feature space of the test data, effectively distinguishing between positive and negative samples by “spreading them

apart” in the feature space. Notably, as the number of layers increases, the positive and negative samples in the feature space become more widely separated, while the clusters formed by similar data become tighter and more organized. This demonstrates the model’s effective learning process in classification tasks.

Fig. 6 illustrates the model’s learning progress at the final layer during the fine-tuning phase as the number of training epochs increases. As training progresses, the feature distribution of positive and negative samples learned by the model becomes increasingly distinct, with a clearer separation between the classes. This indicates that the model’s learning process for downstream tasks is focused on enhancing its ability to differentiate between different categories of data in the feature space, thereby enabling the subsequent linear classification layer to perform the classification task effectively.

To validate the robustness of GEPL, we introduced Gaussian noise to the test data and visualized the feature distribution of the noisy data in the model’s final layer. As illustrated in Fig. 7, even in the presence of noise, the feature distribution in the final layer maintains a high level of distinguishability. Data points from different classes remain clearly grouped, forming relatively independent clusters with distinct boundaries and minimal overlap. This demonstrates that the GEPL model exhibits strong robustness in feature extraction and representation, effectively resisting the interference of Gaussian noise while accurately capturing the essential features of the data. Consequently, this provides a reliable foundation for subsequent classification and other tasks.

Furthermore, we eliminated the graph prompt adjustment for the input data and directly input the EEG graphs from the downstream dataset into the model, rather than modifying them into prompt graphs beforehand. This was taken to emphasize the impact of the graph prompt on the distribution of the feature space. Fig. 8 shows the feature space distribution of the test data in the final layer of the model after the removal of the graph prompt. It isevidentthat,intheabsenceofthegraphprompt,thefeaturedistribution of positive and negative samples becomes chaotic, with indistinct boundaries between clusters and considerable overlap. This suggests that the graph prompt is essential for adjusting the input data, allowing the model to more effectively learn feature representations that are beneficial for classification and to achieve a more optimal sample distribution within the feature space.

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

CUI et al.: “PRE-TRAIN, PROMPT” FRAMEWORK TO BOOST GRAPH NEURAL NETWORKS PERFORMANCE IN EEG ANALYSIS

4857

**==> picture [451 x 301] intentionally omitted <==**

Fig. 5. The distribution of test data within the feature space across different layers. Blue data points represent negative samples (healthy subjects), while red data points indicate positive samples (patients with sudden hearing loss, Alzheimer’s disease, or epilepsy).

## IV. DISCUSSION

GNNs can effectively capture and utilize spatial and functional connectivity information in EEG data, leading to widespread applications of GNN-based methods for EEG analysis [7], [40], [45], [68]. However, the common issue of data scarcity in EEG datasets hinders the effective training of GNN models, adversely affecting their performance in medical tasks. Some studies have employed transfer learning to mitigate the scarcity of EEG data [10], but ensuring model generalization and effective knowledge transfer still poses significant challenges. Therefore, our work aims to improve the performance of GNNs on data-scarce EEG datasets through graph prompt learning. We introduce an efficient, highly adaptable “pre-train, prompt” framework, termed GNN-based EEG Prompt Learning (GEPL), enabling transfer learning between other EEG datasets and target EEG datasets to mitigate the poor performance of GNNs due to target EEG data scarcity. Our framework ultimately outperforms traditional transfer learning methods in various target datasets.

To date, there have been no research efforts focused on addressing the data scarcity issue specifically encountered by GNN-based EEG analysis methods. This research, therefore, represents a pioneering attempt to address this critical gap in the field. Compared to recent studies employing transfer learning to alleviate limitations in EEG data [51], [69], [70], [71], our method, GEPL, is uniquely designed for EEG data structured as graphs, making it highly adaptable to different types of GNNs. Furthermore, while most transfer learning approaches in EEG research tend to focus on transferring knowledge within

closely related EEG domains, the generalizability of these methods to EEG datasets derived from disparate sources remains uncertain [16], [17]. Differences in data acquisition methods, recording devices, and population samples can present substantial challenges for traditional transfer learning techniques when applied to EEG datasets beyond their original domains. In contrast, our study extends the scope of transfer learning by exploring cross-domain EEG applications, thereby assessing the effectiveness and robustness of GEPL in contexts where EEG data originate from diverse sources. This cross-domain capability provides GEPL with a significant advantage, enabling it to handle a broader range of EEG datasets while maintaining high performance. Additionally, rather than following the conventional “pre-train, fine-tune” paradigm prevalent in EEG studies, GEPL introduces an innovative “pre-train, prompt” framework. This approach not only achieves superior performance metrics but also does so with reduced computational demands. Prompt learning is particularly suitable for scenarios where EEG data are scarce, offering a promising avenue for exploration in this area of research.

From a clinical perspective, Section III-H delves into the interpretability of GEPL, illustrating that, in classification tasks spanning multiple target datasets, the model’s high-attention regions consistently correspond with known disease mechanisms. This alignment with clinical findings not only reinforces GEPL’s reliability but also underscores its potential as a diagnostic support tool in resource-limited settings. We believe that this interpretability is crucial for clinicians. An interpretable model can show which biomarkers or clinical features play a key

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 7, JULY 2025

4858

**==> picture [452 x 301] intentionally omitted <==**

Fig. 6. The distribution of test data within the feature space of the last layer for different training epochs. Blue data points represent negative samples (healthy subjects), while red data points indicate positive samples (patients with sudden hearing loss, Alzheimer’s disease, or epilepsy). The rightmost one is the distribution of the trained model.

**==> picture [364 x 99] intentionally omitted <==**

Fig. 7. The distribution of test data with Gaussian noise within the feature space of the last layer. Blue data points represent negative samples (healthy subjects), while red data points indicate positive samples (patients with sudden hearing loss, Alzheimer’s disease, or epilepsy). The rightmost one is the distribution of the trained model.

**==> picture [364 x 99] intentionally omitted <==**

Fig. 8. The distribution of test data without graph prompt within the feature space of the last layer. Blue data points represent negative samples (healthy subjects), while red data points indicate positive samples (patients with sudden hearing loss, Alzheimer’s disease, or epilepsy). The rightmost one is the distribution of the trained model.

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

CUI et al.: “PRE-TRAIN, PROMPT” FRAMEWORK TO BOOST GRAPH NEURAL NETWORKS PERFORMANCE IN EEG ANALYSIS

4859

role in the diagnosis or prediction process, which is essential for determining whether the model relies on scientifically reasonable factors. In real-world clinical applications, GEPL improves the performance of GNNs with minimal data, paving a promisingpathforuseinsettingswhereextensivedatacollection is challenging, such as in vulnerable patient groups or when collection costs are high. By reducing the dependency on large datasets, GEPL offers a pathway for more accessible, costeffective machine learning solutions in healthcare. Furthermore, GEPL is a general method that can offer insights and inspiration for few-shot learning and data scarcity issues in other fields, such as electrocardiograms (ECG) [72] and electromyograms (EMG) [73], which can also be represented as graph-structured medical data. We hope that GEPL can serve as a robust deep learning framework to advance EEG data analysis, enabling researchers to achieve satisfactory results even under data-scarce conditions by leveraging the “pre-train, prompt” framework.

Limitations should be acknowledged in this study. Primarily focusing on EEG data from the frequency domain, this study opens several intriguing future directions, such as exploring the integration of temporal dynamics into the graph prompt learning framework. This would enhance the capture of timesensitive patterns relevant to dynamic clinical tasks, such as seizure prediction and mental workload assessment, or improve backbone models and graph construction for better applicability in time-related predictive clinical tasks.

## V. CONCLUSION

In this study, we proposed a “pre-train, prompt” framework in graph neural networks (GNNs) for EEG analysis, named GEPL. This is a transfer learning method that aims to improve the performance of GNNs when EEG data is scarce. The framework uses contrastive learning for unsupervised pre-training, which does not rely on labeled data, allowing us to effectively use EEG data from other domains to pre-train the model. Then, we use graph prompt learning to achieve efficient fine-tuning in the target EEG dataset, improving the performance of the model. The study experimentally verified the effectiveness and low computational complexity of the framework on a variety of EEGdatasetsandperformedwellinthetasksofdetectingsudden hearing loss, Alzheimer’s disease, and epilepsy. In addition, GEPL is also clinically interpretable.

## REFERENCES

- [1] M. Soufineyestani, D. Dowling, and A. Khan, “Electroencephalography (EEG) technology applications and available devices,” _Appl. Sci._ , vol. 10, no. 21, 2020, Art. no. 7453.

- [2] M. X. Cohen, “Where does EEG come from and what does it mean?,” _Trends Neurosciences_ , vol. 40, no. 4, pp. 208–218, 2017.

- [3] C.-D. Wang et al., “Cross-subject tinnitus diagnosis based on multi-band eeg contrastive representation learning,” _IEEE J. Biomed. Health Inform._ , vol. 27, no. 7, pp. 3187–3197, Jul. 2023.

- [4] L. M. Ribas, F. T. Rocha, N. R. S. Ortega, A. F. da Rocha, and E. Massad, “Brain activity and medical diagnosis: An EEG study,” _BMC Neurosci._ , vol. 14, pp. 1–15, 2013.

- [5] C. Beppi, I. R. Violante, G. Scott, and S. Sandrone, “EEG, MEG and neuromodulatory approaches to explore cognition: Current status and future directions,” _Brain Cogn._ , vol. 148, 2021, Art. no. 105677.

- [6] N. Hooda and N. Kumar, “Scrutinizing different EEG-based mechanisms for motor control and rehabilitation of lower limb disabilities,” _Neurosci. Biomed. Eng._ , vol. 5, no. 1, pp. 50–58, 2017.

- [7] M. Graña and I. Morais-Quilez, “A review of graph neural networks for electroencephalography data analysis,” _Neurocomputing_ , vol. 562, 2023, Art. no. 126901.

- [8] H. Tang et al., “Multi-domain based dynamic graph representation learning for EEG emotion recognition,” _IEEE J. Biomed. Health Inform._ , vol. 28, no. 9, pp. 5227–5238, Sep. 2024.

- [9] H. Chang, B. Liu, Y. Zong, C. Lu, and X. Wang, “EEG-based Parkinson’s disease recognition via attention-based sparse graph convolutional neural network,” _IEEE J. Biomed. Health Inform._ , vol. 27, no. 11, pp. 5216–5224, Nov. 2023.

- [10] Z. Wan, R. Yang, M. Huang, N. Zeng, and X. Liu, “A review on transfer learning in EEG signal analysis,” _Neurocomputing_ , vol. 421, pp. 1–14, 2021.

- [11] E. Lashgari, D. Liang, and U. Maoz, “Data augmentation for deeplearning-based electroencephalography,” _J. Neurosci. Methods_ , vol. 346, 2020, Art. no. 108885.

- [12] C. He, J. Liu, Y. Zhu, and W. Du, “Data augmentation for deep neural networks model in EEG classification task: A review,” _Front. Hum. Neurosci._ , vol. 15, 2021, Art. no. 765525.

- [13] R. Li, L. Wang, P. N. Suganthan, and O. Sourina, “Sample-based data augmentation based on electroencephalogram intrinsic characteristics,” _IEEE J. Biomed. Health Inform._ , vol. 26, no. 10, pp. 4996–5003, Oct. 2022.

- [14] C. Sun, J. Fan, C. Chen, W. Li, and W. Chen, “A two-stage neural network for sleep stage classification based on feature learning, sequence learning, and data augmentation,” _IEEE Access_ , vol. 7, pp. 109386–109397, 2019.

- [15] D. Wu, Y. Xu, and B.-L. Lu, “Transfer learning for EEG-based brain– computer interfaces: A review of progress made since 2016,” _IEEE Trans. Cogn. Develop. Syst._ , vol. 14, no. 1, pp. 4–19, Mar. 2022.

- [16] Y.-P. Lin, “Constructing a personalized cross-day EEG-based emotionclassification model using transfer learning,” _IEEE J. Biomed. Health Inform._ , vol. 24, no. 5, pp. 1255–1264, May 2020.

- [17] P.-Y. Jeng, C.-S. Wei, T.-P. Jung, and L.-C. Wang, “Low-dimensional subject representation-based transfer learning in EEG decoding,” _IEEE J. Biomed. Health Inform._ , vol. 25, no. 6, pp. 1915–1925, Jun. 2021.

- [18] P. Nejedly et al., “Intracerebral EEG artifact identification using convolutional neural networks,” _Neuroinformatics_ , vol. 17, pp. 225–234, 2019.

- [19] F. Zhang, H. Wu, and Y. Guo, “Semi-supervised multi-source transfer learning for cross-subject EEG motor imagery classification,” _Med. Biol. Eng. Comput._ , vol. 62, pp. 1655–1672, 2024.

- [20] W. Jin et al., “Self-supervised learning on graphs: Deep insights and new direction,” 2020, _arXiv:2006.10141_ .

- [21] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, “All in one: Multi-task prompting for graph neural networks,” in _Proc. 29th ACM SIGKDD Conf. Knowl. Discov. Data Mining_ , 2023, pp. 2120–2131.

- [22] F. Zhou and C. Cao, “Overcoming catastrophic forgetting in graph neural networks with experience replay,” in _Proc. AAAI Conf. Artif. Intell._ , 2021, pp. 4714–4722.

- [23] H. Liu, Y. Yang, and X. Wang, “Overcoming catastrophic forgetting in graph neural networks,” in _Proc. AAAI Conf. Artif. Intell._ , 2021, pp. 8653–8661.

- [24] C. Zhang et al., “Few-shot learning on graphs,” in _Proc. 31st IJCAI Conf. Artif. Intell._ , 2022, pp. 5662–5669.

- [25] P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig, “Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing,” _ACM Comput. Surv._ , vol. 55, no. 9, pp. 1–35, 2023.

- [26] K. Chowdhary, “Natural language processing,” _Fund. Artif. Intell._ , pp. 603–649, 2020.

- [27] M.Sun,K.Zhou,X.He,Y.Wang,andX.Wang,“GPPT:Graphpre-training andprompttuningtogeneralizegraphneuralnetworks,”in _Proc.28thACM SIGKDD Conf. Knowl. Discov. Data Mining_ , 2022, pp. 1717–1727.

- [28] Y. Wang, Y. Xiong, X. Wu, X. Sun, J. Zhang, and G. Zheng, “Ddiprompt: Drug-drug interaction event prediction based on graph prompt learning,” in _Proc. 33rd ACM Int. Conf. Inf. Knowl. Manag._ , 2024, pp. 2431–2441.

- [29] X. Yu, Y. Fang, Z. Liu, and X. Zhang, “HGPrompt: Bridging homogeneous and heterogeneous graphs for few-shot prompt learning,” in _Proc. AAAI Conf. Artif. Intell._ , 2024, pp. 16578–16586.

- [30] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen, “Graph contrastive learning with augmentations,” in _Proc. Adv. Neural Inf. Process. Syst._ , 2020, pp. 5812–5823.

- [31] N. K. Al-Qazzaz, S. H. B. M. Ali, S. A. Ahmad, K. Chellappan, M. S. Islam, and J. Escudero, “Role of EEG as biomarker in the early detection and classification of dementia,” _Sci. World J._ , vol. 2014, no. 1, 2014, Art. no. 906038.

- [32] S. N. Merchant, M. L. Durand, and J. C. Adams, “Sudden deafness: Is it viral?,” _ORL_ , vol. 70, no. 1, pp. 52–62, 2008.

- [33] U. R. Acharya, S.V. Sree, G. Swapna, R. J. Martis, and J. S. Suri, “Automated EEG analysis of epilepsy: A review,” _Knowl.-Based Syst._ , vol. 45, pp. 147–165, 2013.

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 29, NO. 7, JULY 2025

4860

- [34] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” 2018, _arXiv:1807.03748_ .

- [35] T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen, “Universal prompt tuning for graph neural networks,” in _Proc. Adv. Neural Inf. Process. Syst._ , 2024, pp. 52464–52489.

- [36] S. Koelstra et al., “DEAP: A database for emotion analysis; using physiological signals,” _IEEE Trans. Affect. Comput._ , vol. 3, no. 1, pp. 18–31, Jan.–Mar. 2012.

- [37] A. Miltiadous et al., “A dataset of EEG recordings from: Alzheimer’s disease, frontotemporal dementia and healthy subjects,” _OpenNeuro_ , vol. 1, p. 88, 2023.

- [38] I. Tasci et al., “Epilepsy detection in 121 patient populations using hypercube pattern from EEG signals,” _Inf. Fusion_ , vol. 96, pp. 252–268, 2023.

- [39] P. Duhamel and M. Vetterli, “Fast fourier transforms: A tutorial review and a state of the art,” _Signal Process._ , vol. 19, no. 4, pp. 259–299, 1990.

- [40] S. Tang et al., “Self-supervised graph neural networks for improved electroencephalographic seizure analysis,” in _Proc. Int. Conf. Learn. Representations_ , 2022.

- [41] P. Schober, C. Boer, and L. A. Schwarte, “Correlation coefficients: Appropriate use and interpretation,” _Anesth. Analg._ , vol. 126, no. 5, pp. 1763–1768, 2018.

- [42] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in _Proc. Int. Conf. Learn. Representations_ , 2017.

- [43] K. Polat and S. Güne¸s, “Classification of epileptiform EEG using a hybrid system based on decision tree classifier and fast fourier transform,” _Appl. Math. Comput._ , vol. 187, no. 2, pp. 1017–1026, 2007.

- [44] C. J. Burges, “A tutorial on support vector machines for pattern recognition,” _Data Mining Knowl. Discov._ , vol. 2, no. 2, pp. 121–167, 1998.

- [45] Z. Li, K. Hwang, K. Li, J. Wu, and T. Ji, “Graph-generative neural network for EEG-based epileptic seizure detection via discovery of dynamic brain functional connectivity,” _Sci. Rep._ , vol. 12, no. 1, 2022, Art. no. 18998.

- [46] A. Demir, T. Koike-Akino, Y. Wang, M. Haruna, and D. Erdogmus, “EEGGNN: Graph neural networks for classification of electroencephalogram (EEG) signals,” in _Proc. 43rd Annu. Int. Conf. IEEE Eng. Med. Biol. Soc._ , 2021, pp. 1061–1067.

- [47] D. Klepl, F. He, M. Wu, D. J. Blackburn, and P. Sarrigiannis, “EEG-based graph neural network classification of Alzheimer’s disease: An empirical evaluation of functional connectivity methods,” _IEEE Trans. Neural Syst. Rehabil. Eng._ , vol. 30, pp. 2651–2660, 2022.

- [48] S. An, S. Kim, P. Chikontwe, and S. H. Park, “Dual attention relation network with fine-tuning for few-shot EEG motor imagery classification,” _IEEE Trans. Neural Netw. Learn. Syst._ , vol. 35, no. 11, pp. 15479–15493, Nov. 2024.

- [49] X. Jia, Y. Song, and L. Xie, “Excellent fine-tuning: From specific-subject classification to cross-task classification for motor imagery,” _Biomed. Signal Process. Control_ , vol. 79, 2023, Art. no. 104051.

- [50] R. Zhang, Q. Zong, L. Dou, X. Zhao, Y. Tang, and Z. Li, “Hybrid deep neural network using transfer learning for EEG motor imagery decoding,” _Biomed. Signal Process. Control_ , vol. 63, 2021, Art. no. 102144.

- [51] K. Zhang, N. Robinson, S.-W. Lee, and C. Guan, “Adaptive transfer learning for EEG motor imagery classification with deep convolutional neural network,” _Neural Netw._ , vol. 136, pp. 1–10, 2021.

- [52] G. Xu et al., “A deep transfer convolutional neural network framework for EEG signal classification,” _IEEE Access_ , vol. 7, pp. 112767–112776, 2019.

- [53] P. Veliˇckovi´c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, “Graph attention networks,” in _Proc. Int. Conf. Learn. Representations_ , 2018.

 - [55] W. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in _Proc. Adv. Neural Inf. Process. Syst._ , 2017, pp. 1025–1035.

 - [56] K. Simonyan, A. Vedaldi, and A. Zisserman, “Deep inside convolutional networks: Visualising image classification models and saliency maps,” in _Proc. Int. Conf. Learn. Representations_ , 2014.

 - [57] J. Chen et al., “Altered brain activity and functional connectivity in unilateral sudden sensorineural hearing loss,” _Neural Plast._ , vol. 2020, no. 1, 2020, Art. no. 9460364.

 - [58] Z. Li, “Analysis of changes in brain region and connectomics in the acute stage of sudden sensorineural hearing loss in the resting state via functional magnetic resonance imaging,” _Concepts Magn. Reson. Part A_ , vol. 2023, no. 1, 2023, Art. no. 7007209.

 - [59] J. Defenderfer, A. Kerr-German, M. Hedrick, and A. T. Buss, “Investigating the role of temporal lobe activation in speech perception accuracy with normal hearing adults: An event-related fnirs study,” _Neuropsychologia_ , vol. 106, pp. 31–41, 2017.

 - [60] H. I. Jacobs, M. P. Van Boxtel, J. Jolles, F. R. Verhey, and H. B. Uylings, “Parietal cortex matters in alzheimer’s disease: An overview of structural, functional and metabolic findings,” _Neurosci. Biobehavioral Rev._ , vol. 36, no. 1, pp. 297–309, 2012.

 - [61] T. Thientunyakit et al., “Relationship between F-18 florbetapir uptake in occipital lobe and neurocognitive performance in alzheimer’s disease,” _Japanese J. Radiol._ , vol. 39, no. 10, pp. 984–993, 2021.

 - [62] J. K. Johnson, E. Head, R. Kim, A. Starr, and C. W. Cotman, “Clinical and pathological evidence for a frontal variant of alzheimer disease,” _Arch. Neurol._ , vol. 56, no. 10, pp. 1233–1239, 1999.

 - [63] P. Beleza and J. Pinho, “Frontal lobe epilepsy,” _J. Clin. Neurosci._ , vol. 18, no. 5, pp. 593–600, 2011.

 - [64] J. Stretton and P. Thompson, “Frontal lobe function in temporal lobe epilepsy,” _Epilepsy Res._ , vol. 98, no. 1, pp. 1–13, 2012.

 - [65] J. Engel Jr., “Introduction to temporal lobe epilepsy,” _Epilepsy Res._ , vol. 26, no. 1, pp. 141–150, 1996.

 - [66] M. Duchowny, P. Jayakar, T. Resnick, B. Levin, and L. Alvarez, “Posterior temporal epilepsy: Electroclinical features,” _Ann. Neurol.: Official J. Amer. Neurological Assoc. Child Neurol. Soc._ , vol. 35, no. 4, pp. 427–431, 1994.

 - [67] M. T. Islam et al., “Revealing hidden patterns in deep neural network feature space continuum via manifold learning,” _Nature Commun._ , vol. 14, no. 1, 2023, Art. no. 8506.

 - [68] Y. Ding, N. Robinson, C. Tong, Q. Zeng, and C. Guan, “LGGNet: Learning from local-global-graph representations for brain–computer interface,” _IEEE Trans. Neural Netw. Learn. Syst._ , vol. 35, no. 7, pp. 9773–9786, Jul. 2024.

 - [69] J. Li, S. Qiu, Y.-Y. Shen, C.-L. Liu, and H. He, “Multisource transfer learning for cross-subject EEG emotion recognition,” _IEEE Trans. Cybern._ , vol. 50, no. 7, pp. 3281–3293, Jul. 2020.

 - [70] F. Fahimi, Z. Zhang, W. B. Goh, T.-S. Lee, K. K. Ang, and C. Guan, “Inter-subject transfer learning with an end-to-end deep convolutional neural network for EEG-based BCI,” _J. Neural Eng._ , vol. 16, no. 2, 2019, Art. no. 026007.

 - [71] Z. Khademi, F. Ebrahimi, and H. M. Kordy, “A transfer learning-based CNN and LSTM hybrid deep learning model to classify motor imagery EEG signals,” _Comput. Biol. Med._ , vol. 143, 2022, Art. no. 105288.

 - [72] A. Mincholé, J. Camps, A. Lyon, and B. Rodríguez, “Machine learning in the electrocardiogram,” _J. Electrocardiol._ , vol. 57, pp. S61–S64, 2019.

 - [73] D. Wu, J. Yang, and M. Sawan, “Transfer learning on electromyography (EMG) tasks: Approaches and beyond,” _IEEE Trans. Neural Syst. Rehabil. Eng._ , vol. 31, pp. 3015–3034, 2023.

- [54] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph neural networks?,” in _Proc. Int. Conf. Learn. Representations_ , 2019.

Authorized licensed use limited to: Chung Yuan Christian University. Downloaded on September 04,2025 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.