# GPU-Network-Testing
Statistics displaying the performance of a neural network on the CPU vs the GPU. A network is stored as three floating-point arrays representing the nodes, weights, and biases. The **length** of a network tells how many layers there are, while the **width** tells how many rows of nodes are in a single layer. 

The statistics shown below display how the performance changes when modifying the length and width of a network on both the GPU and the CPU. The code that generated this data is provided in [Network.cu](https://github.com/justinmgarrigus/GPU-Network-Testing/blob/main/Network.cu); code is written in Cuda for C++ using Visual Studio 2019, and was executed on a machine with an NVIDIA GeForce RTX 2080 Super graphics card. 

## Feedforward 
| ![Small width, increasing length](https://github.com/justinmgarrigus/GPU-Network-Testing/blob/main/Data/Images/Small%20width%20increasing%20length.png) | 
|:--:| 
| *The length of a network is plotted against its execution time. These networks have a small width (10 nodes) with lengths between 2 and 50. A highlighted point shows that a length of 42 runs at 0.267 seconds on the CPU and 19.11 seconds on the GPU. The GPU takes much longer to run than the CPU does when the width of the network is low.* |

| ![Large width, increasing length](https://github.com/justinmgarrigus/GPU-Network-Testing/blob/main/Data/Images/Large%20width%20increasing%20length.png) | 
|:--:| 
| *The length of a network is plotted against its execution time. These networks have a large width (10000 nodes) with lengths between 2 and 8. A highlighted point shows that a length of 7 runs at 86.689 seconds on the CPU and 1.222 seconds on the GPU. The GPU takes much shorter to run than the CPU does when the width of the network is high.* |

| ![Increasing width, small length](https://github.com/justinmgarrigus/GPU-Network-Testing/blob/main/Data/Images/Increasing%20width%20small%20length.png) | 
|:--:| 
| *The width of a network is plotted against its execution time. These networks have a small length (5 nodes) with widths between 2 and 7000. A highlighted point shows that a width of 5700 runs at 16.025 seconds on the CPU and 0.577 seconds on the GPU. The GPU takes much shorter to run than the CPU does when the width of the network is high, regardless of the length* |

![Increasing width, large length](https://github.com/justinmgarrigus/GPU-Network-Testing/blob/main/Data/Images/Increasing%20width%20large%20length.png) | 
|:--:| 
| *The width of a network is plotted against its execution time. These networks have a large length (20 nodes) with widths between 2 and 7000. A highlighted point shows that a width of 5000 runs at 16.067 seconds on the CPU and 0.578 seconds on the GPU. The results are consistent with the previous test; length does not affect execution time for a GPU-network as much as width does.* |

The CSV data plots, as well as the images themselves, can be found in the [Data](https://github.com/justinmgarrigus/GPU-Network-Testing/tree/main/Data) folder. 
