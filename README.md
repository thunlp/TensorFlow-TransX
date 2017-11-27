# TensorFlow-TransX

The implementation of TransE [1], TransH [2], TransR [3], TransD [4] for knowledge representation learning (KRL). The overall framework is based on TensorFlow. We use C++ to implement some underlying operations such as data preprocessing and negative sampling. For each specific model, it is implemented by TensorFlow with Python interfaces so that there is a convenient platform to run models on GPUs. 

These codes will be gradually integrated into the new framework [[OpenKE]](https://github.com/thunlp/openke).

# Customizing Your Own Model

If you have a new idea and need to implement its code, you just need to change Python interfaces for your customized model. Read these codes, you will find that to change the class TransXModel will meet your needs.

# Evaluation Results

More results about models can be found in ("https://github.com/thunlp/KB2E").

# Data

Datasets are required in the following format, containing three files:

triple2id.txt: training file, the first line is the number of triples for training. Then the follow lines are all in the format (e1, e2, rel).

entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

You can download FB15K and WN18 from [[Download]](https://github.com/thunlp/Fast-TransX/tree/master/data), and the more datasets can also be found in ("https://github.com/thunlp/KB2E").

# Compile

bash make.sh

# Train

To train models based on random initialization:

1. Change class Config in transX.py

		class Config(object):
	
			def __init__(self):
				...
				lib.setInPath("your training data path...")
				self.testFlag = False
				self.loadFromData = False
				...

2. python transX.py

To train models based on pretrained results:

1. Change class Config in transX.py

		class Config(object):
	
			def __init__(self):
				...
				lib.setInPath("your training data path...")
				self.testFlag = False
				self.loadFromData = True
				...

2. python transX.py

# Test

To test your models:

1. Change class Config in transX.py
	
		class Config(object):

			def __init__(self):
				...
				test_lib.setInPath("your testing data path...")
				self.testFlag = True
				self.loadFromData = True
				...

2. python transX.py



# Citation

If you use the code, please kindly cite the papers listed in our reference.

# Reference

[1] Bordes, Antoine, et al. Translating embeddings for modeling multi-relational data. Proceedings of NIPS, 2013.

[2]	Zhen Wang, Jianwen Zhang, et al. Knowledge Graph Embedding by Translating on Hyperplanes. Proceedings of AAAI, 2014.

[3] Yankai Lin, Zhiyuan Liu, et al. Learning Entity and Relation Embeddings for Knowledge Graph Completion. Proceedings of AAAI, 2015.

[4] Guoliang Ji, Shizhu He, et al. Knowledge Graph Embedding via Dynamic Mapping Matrix. Proceedings of ACL, 2015.
