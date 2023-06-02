# Models proposed - Session 2 - _De-novo_ molecular desing

1. Gradient-free models

    1. A graph-based genetic algorithm and generative model/Monte Carlo tree search for the exploration of chemical space \
    Genetic algorithm using MCTS on graphs, atom-based
    > Jensen, J. H. (2019). A graph-based genetic algorithm and generative model/Monte Carlo tree search for the exploration of chemical space. Chemical science, 10(12), 3567-3572.  
    MolSearch implementation: https://github.com/illidanlab/MolSearch
  
    2. AutoGrow4: an open-source genetic algorithm for de novo drug design and lead optimization \
    Genetic model that starts from fragments, reaction-based
    > Spiegel, J. O., & Durrant, J. D. (2020). AutoGrow4: an open-source genetic algorithm for de novo drug design and lead optimization. Journal of cheminformatics, 12(1), 1-16.

    3. CReM: chemically reasonable mutations framework for structure generation \
    Fragment-based structure generation of chemically valid structures by design
    > Polishchuk, P. (2020). CReM: chemically reasonable mutations framework for structure generation. Journal of Cheminformatics, 12(1), 1-18.

    4. EvoMol: a flexible and interpretable evolutionary algorithm for unbiased de novo molecular generation \
    Evolutionary algorithm to sequentially build molecular graphs
    > Leguy, J., Cauchy, T., Glavatskikh, M., Duval, B., & Da Mota, B. (2020). EvoMol: a flexible and interpretable evolutionary algorithm for unbiased de novo molecular generation. Journal of cheminformatics, 12(1), 1-19.


2. Gradient-based models

    1. Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules ($\star$) \
    VAE + GP + Bayesian optimization using SMILES
    > Gómez-Bombarelli, Rafael, et al. "Automatic chemical design using a data-driven continuous representation of molecules." ACS central science 4.2 (2018): 268-276.

    2. Hierarchical Generation of Molecular Graphs using Structural Motifs \
    Hierarchical encoder-decoder model with message passing networks using graphs 
    > Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. "Hierarchical generation of molecular graphs using structural motifs." In International conference on machine learning, pp. 4839-4848. PMLR, 2020.

    3. Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2 ($\star$) \
    Protein-ligand de-novo model, VAE + RL approach using different text-based encoding 
    > Born, Jannis, et al. "Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2." Machine Learning: Science and Technology 2.2 (2021): 025024.

    4. Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks \
    RNN-based model using SMILES
    > Segler, M. H., Kogej, T., Tyrchan, C., & Waller, M. P. (2018). Generating focused molecule libraries for drug discovery with recurrent neural networks. ACS central science, 4(1), 120-131.
    (**code unclear**)
  
    5. Generative molecular design in low data regimes \
    LSTM model for low-data regimes using SMILES
    > Moret, M., Friedrich, L., Grisoni, F., Merk, D., & Schneider, G. (2020). Generative molecular design in low data regimes. Nature Machine Intelligence, 2(3), 171-180.

    6. Junction Tree Variational Autoencoder for Molecular Graph Generation \
    Fragment-based VAE model
    > Jin, W., Barzilay, R., & Jaakkola, T. (2018, July). Junction tree variational autoencoder for molecular graph generation. In International conference on machine learning (pp. 2323-2332). PMLR.

    7. Barking up the right tree: an approach to search over molecule synthesis DAGs \
    Reaction-based model that uses deep networks and outputs DAGs for molecule synthesis
    > Bradshaw, J., Paige, B., Kusner, M. J., Segler, M., & Hernández-Lobato, J. M. (2020). Barking up the right tree: an approach to search over molecule synthesis dags. Advances in neural information processing systems, 33, 6852-6866. 

    8. Equivariant Diffusion for Molecule Generation in 3D ($\star$) \
    Diffusion model for targeted generation with 3D representations
    > Hoogeboom, Emiel, et al. "Equivariant diffusion for molecule generation in 3d." International Conference on Machine Learning. PMLR, 2022.

    9. GraphAF: A Flow-Based Autorregresive Model for Molecular Graph Generation \
    Normalizing-flow model that enables exact estimations
    > Shi, C., Xu, M., Zhu, Z., Zhang, W., Zhang, M., & Tang, J. (2020). Graphaf: a flow-based autoregressive model for molecular graph generation. arXiv preprint arXiv:2001.09382.