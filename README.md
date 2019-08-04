# Conv-KNRM
This paper presents Conv-KNRM, a Convolutional Kernel-based Neural Ranking Model that models n-gram soft matches for ad-hoc search. Instead of exact matching query and document n-grams, Conv-KNRM uses Convolutional Neural Networks to represent ngrams of various lengths and soft matches them in a unified embedding space. the n-gram soft matches are then utilized by the kernel pooling and learning-to-rank layers to generate the final ranking score. Conv-KNRM can be learned end-to-end and fully optimized from user feedback. The learned model’s generalizability is investigated by testing how well it performs in a related domain with small amounts of training data. Experiments on English search logs, Chinese search logs, and TREC Web track tasks demonstrated consistent advantages of Conv-KNRM over prior neural IR methods and feature-based methods.


[Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search (WSDM-2018)](http://www.cs.cmu.edu/~./callan/Papers/wsdm18-zhuyun-dai.pdf "Conv-KNRM")


![image](https://github.com/jyy0553/Conv-KNRM/tree/master/IMG)
