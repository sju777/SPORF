#include <queue>
#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <sstream>
#include "improv8.h"

namespace {

	inline int returnClassPrediction(std::vector<int>& classTotals, int numClasses){
		int predictMaxValue = -1;
		int predictMaxClass = -1;
		for(int m=0; m<numClasses; m++){
			if(classTotals[m] > predictMaxValue){
				predictMaxClass = m;
				predictMaxValue = classTotals[m];
			}
		}
		return predictMaxClass;
	}

	class repack 
	{
		public:
			int currentNode=-1;
			padNodeStat* tree;
			padNode* realTree;
			int repackTree(int workingNode);
			repack( padNodeStat* tTree, padNode* rTree): tree(tTree), realTree(rTree) {};
	};

	int repack::repackTree(int workingNode){

		int thisNodesLocation = ++currentNode;

		if(tree[workingNode].isInternalNode()){
			int leftFreq = tree[tree[workingNode].returnLeftNode()].returnFreq();
			int rightFreq = tree[tree[workingNode].returnRightNode()].returnFreq();
			int leftNodeNum;
			int rightNodeNum;
			if(rightFreq > leftFreq){
				rightNodeNum = repackTree(tree[workingNode].returnRightNode());
				leftNodeNum = repackTree(tree[workingNode].returnLeftNode());
			}else{
				leftNodeNum = repackTree(tree[workingNode].returnLeftNode());
				rightNodeNum = repackTree(tree[workingNode].returnRightNode());
			}

			realTree[thisNodesLocation].setNode(tree[workingNode].returnCutValue(),
					tree[workingNode].returnFeature(),
					leftNodeNum,
					rightNodeNum
					);
		}else{
			realTree[thisNodesLocation].setNode(tree[workingNode].returnCutValue(),
					tree[workingNode].returnFeature(),
					tree[workingNode].returnLeftNode(),
					tree[workingNode].returnRightNode());
		}
		return(thisNodesLocation);
	}
} //namespace

improv8::improv8(const std::string& forestFileName){
	std::ifstream infile (forestFileName, std::ofstream::binary);

	infile.read((char*)&numOfBins, sizeof(int));
	infile.read((char*)&numTreesInForest, sizeof(int));
	infile.read((char*)&totalNumberOfNodes, sizeof(int));
	infile.read((char*)&numOfClasses, sizeof(int));
	infile.read((char*)&debugModeOn, sizeof(bool));
	infile.read((char*)&showAllResults, sizeof(bool));

	for(int i = 0 ; i < numOfBins; i++){
		forestRoots.emplace_back(infile);
	}

	int testEOF;
	if(infile >> testEOF){
		Rprintf("not end of file \n");
	}
	infile.close();
}

improv8::improv8(const std::string& forestCSVFileName, int source, const inferenceSamples& observations, int numberBins, int depthIntertwined){
	if(source == 1){
		std::ifstream fin(forestCSVFileName.c_str());
		int numValuesForTree;
		int numInnerNodes;
		int numInnerNodesActual;
		totalNumberOfNodes = 0;
		int numLeafNodesInForest = 0;
		double num;
		std::vector<std::vector<padNodeStat> > tempForestRoots;
		numOfBins = numberBins;

		//First number in csv is the number of trees
		fin >> num;
		numTreesInForest = (int)num;
		//Second number in csv is the number of classes
		fin >> num;
		numOfClasses = (int)num;

		tempForestRoots.resize(numTreesInForest);
		std::vector<int> numNodesInTree(numTreesInForest);

		//Create each tree one at a time.
		for(int i =0; i < numTreesInForest; i++){
			std::vector<double> numbers;

			//First number in each tree is the number of nodes 
			//in the current tree * 2 (map and nodes)
			fin >> num;
			numNodesInTree[i] = (int)num;
			numInnerNodes = (numNodesInTree[i]-1)/2;
			numValuesForTree = numNodesInTree[i]*2+numInnerNodes;
			totalNumberOfNodes += numNodesInTree[i]; 

			if(numNodesInTree[i]%2 == 0){
				Rprintf("num of nodes is even"); 
			}
			//Put all values pertaining to current tree in a vector
			for(int j = 0; j < numValuesForTree; j++){
				fin >> num;
				numbers.push_back(num);  // store the number 
			}
			//Allocate space for this tree
			tempForestRoots[i].resize(numNodesInTree[i]);

			numInnerNodesActual = 0;
			for(int k = 0; k < numNodesInTree[i]; k++){
				if(numbers[k] > 0){
					tempForestRoots[i][k].setNode(numbers[numNodesInTree[i]+numbers[k]-1],
							int(numbers[numNodesInTree[i]*2+numbers[k]-1]),
							int(numbers[k])*2-1,
							int(numbers[k])*2);

					++numInnerNodesActual;
				}else{
					tempForestRoots[i][k].setNode(-1.0,
							int(0),
							int(0),
							int(numbers[numNodesInTree[i]+ numInnerNodes +(-1*numbers[k])-1]));

					++numLeafNodesInForest;
				}
			}

			if(numInnerNodesActual != numInnerNodes ){
				Rprintf("miscalculated inner Nodes: calc-%d, act-%d ", numInnerNodes, numInnerNodesActual);
			}

			if(debugModeOn){
				if(i == 0){
					for(int p = 0; p < numNodesInTree[i]; p++){
						tempForestRoots[i][p].printNode();
					}
				}
			}

			//Print current tree.  Just for checking.
			if(debugModeOn){
				Rprintf("start of tree %d.\n", i);
				for (unsigned int z=0; z<numbers.size(); z++)
					Rprintf("%f\n", numbers[z]);
			}
		}

		//Pull one more float so that eof is TRUE.
		fin >> num;
		if(!fin.eof()){
			Rprintf("csv not exausted");
		}else{
			fin.close();
		}

		int finalTree;
		int startTree=0;
		int binSize = numTreesInForest/numberBins;

		for(int q = 0; q < numberBins; q++){
			startTree = q*binSize;
			finalTree = startTree+binSize;

			if(finalTree > numTreesInForest){
				finalTree = numTreesInForest;
			}
			forestRoots.emplace_back(tempForestRoots, numNodesInTree, startTree, finalTree, depthIntertwined, numOfClasses);
		}
	}         

}

improv8::~improv8(){
}


void improv8::makePredictions(const inferenceSamples& observations){

	std::vector<int> predictions(numOfClasses);
	std::vector<int> currentNode(forestRoots[0].numOfTreesInBin);
	int numberNotInLeaf;
	int  p, k, q;

	for(int i = 0; i < observations.numObservations; i++){

		for( p= 0; p < numOfClasses;++p){
			predictions[p]=0;
		}

		for( k=0; k < numOfBins;++k){
			for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
				currentNode[q] = q;
				__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
			}

			numberNotInLeaf = 1;
			while(numberNotInLeaf > 0){
				numberNotInLeaf = forestRoots[k].numOfTreesInBin;

				for( q=0; q<forestRoots[k].numOfTreesInBin; ++q){

					if(forestRoots[k].bin[currentNode[q]].isInternalNode()){
						currentNode[q] = forestRoots[k].bin[currentNode[q]].nextNode(observations.samplesMatrix[i][forestRoots[k].bin[currentNode[q]].returnFeature()]);
						__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
						continue;
					}
					--numberNotInLeaf;
				}
			}

			for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
				++predictions[forestRoots[k].bin[currentNode[q]].returnRightNode()];
			}

		}

	}
}


void improv8::makePrediction(double* observation, double* preds, int numFeatures, int numObservations, int numCores){

	std::vector<int> predictions(numOfClasses);
	std::vector<int> currentNode(forestRoots[0].numOfTreesInBin);
	int numberNotInLeaf;
	int k, q;
	int observationOffset = 0;

	for(int j = 0; j < numObservations; j++){
		for( k=0; k < numOfBins;k++){

			for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
				currentNode[q] = q;
				__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
			}

			do{
				numberNotInLeaf = forestRoots[k].numOfTreesInBin;

				for( q=0; q<forestRoots[k].numOfTreesInBin; q++){

					if(forestRoots[k].bin[currentNode[q]].isInternalNode()){

						currentNode[q] = forestRoots[k].bin[currentNode[q]].nextNode(observation[forestRoots[k].bin[currentNode[q]].returnFeature()+observationOffset]);

						__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
						continue;
					}
					--numberNotInLeaf;
				}
			}while(numberNotInLeaf > 0);

			for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
				++predictions[forestRoots[k].bin[currentNode[q]].returnRightNode()];
			}
		}
		preds[j] = returnClassPrediction(predictions, numOfClasses);

		observationOffset+=numFeatures;
	}
}


int improv8::makePrediction(std::vector<double>& observation){

	std::vector<int> predictions(numOfClasses);
	std::vector<int> currentNode(forestRoots[0].numOfTreesInBin);
	int numberNotInLeaf;
	int k, q;

	for( k=0; k < numOfBins;k++){

		for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
			currentNode[q] = q;
			__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
		}

		do{
			numberNotInLeaf = forestRoots[k].numOfTreesInBin;

			for( q=0; q<forestRoots[k].numOfTreesInBin; q++){

				if(forestRoots[k].bin[currentNode[q]].isInternalNode()){
					currentNode[q] = forestRoots[k].bin[currentNode[q]].nextNode(observation[forestRoots[k].bin[currentNode[q]].returnFeature()]);
					__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
					continue;
				}
				--numberNotInLeaf;
			}
		}while(numberNotInLeaf > 0);

		for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
			++predictions[forestRoots[k].bin[currentNode[q]].returnRightNode()];
		}
	}
	return returnClassPrediction(predictions, numOfClasses);
}

int improv8::makePrediction(double*& observation){

	std::vector<int> predictions(numOfClasses);
	std::vector<int> currentNode(forestRoots[0].numOfTreesInBin);
	int numberNotInLeaf;
	int k, q;

	for( k=0; k < numOfBins;k++){

		for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
			currentNode[q] = q;
			__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
		}

		do{
			numberNotInLeaf = forestRoots[k].numOfTreesInBin;

			for( q=0; q<forestRoots[k].numOfTreesInBin; q++){

				if(forestRoots[k].bin[currentNode[q]].isInternalNode()){
					currentNode[q] = forestRoots[k].bin[currentNode[q]].nextNode(observation[forestRoots[k].bin[currentNode[q]].returnFeature()]);
					__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
					continue;
				}
				--numberNotInLeaf;
			}
		}while(numberNotInLeaf > 0);

		for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
			++predictions[forestRoots[k].bin[currentNode[q]].returnRightNode()];
		}
	}
	return returnClassPrediction(predictions, numOfClasses);
}


int improv8::makePrediction(double*& observation, int numCores){

	std::vector<int> predictions(numOfClasses);
	std::vector<int> currentNode(forestRoots[0].numOfTreesInBin);
	int numberNotInLeaf;
	int k, q;

	for( k=0; k < numOfBins;k++){

		for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
			currentNode[q] = q;
			__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
		}

		do{
			numberNotInLeaf = forestRoots[k].numOfTreesInBin;

			for( q=0; q<forestRoots[k].numOfTreesInBin; q++){

				if(forestRoots[k].bin[currentNode[q]].isInternalNode()){
					currentNode[q] = forestRoots[k].bin[currentNode[q]].nextNode(observation[forestRoots[k].bin[currentNode[q]].returnFeature()]);
					__builtin_prefetch(&forestRoots[k].bin[currentNode[q]], 0, 3);
					continue;
				}
				--numberNotInLeaf;
			}
		}while(numberNotInLeaf > 0);

		for( q=0; q<forestRoots[k].numOfTreesInBin; q++){
			++predictions[forestRoots[k].bin[currentNode[q]].returnRightNode()];
		}
	}
	return returnClassPrediction(predictions, numOfClasses);
}

void improv8::writeForest(const std::string& forestFileName){
	std::ofstream outfile (forestFileName, std::ofstream::binary);

	outfile.write((char*)&numOfBins, sizeof(int));
	outfile.write((char*)&numTreesInForest, sizeof(int));
	outfile.write((char*)&totalNumberOfNodes, sizeof(int));
	outfile.write((char*)&numOfClasses, sizeof(int));
	outfile.write((char*)&debugModeOn, sizeof(bool));
	outfile.write((char*)&showAllResults, sizeof(bool));
	for(int i = 0 ; i < numOfBins; i++){

		outfile.write((char*)&forestRoots[i].numOfTreesInBin, sizeof(int));
		outfile.write((char*)&forestRoots[i].depth, sizeof(int));
		outfile.write((char*)&forestRoots[i].numOfClasses, sizeof(int));
		outfile.write((char*)&forestRoots[i].numOfNodes, sizeof(int));

		for(int j = 0; j < (forestRoots[i].numOfNodes+forestRoots[i].numOfClasses); j++){

			outfile.write((char*)&forestRoots[i].bin[j].left, sizeof(uint32_t));
			outfile.write((char*)&forestRoots[i].bin[j].feature, sizeof(uint32_t));
			outfile.write((char*)&forestRoots[i].bin[j].cutValue, sizeof(double));
			outfile.write((char*)&forestRoots[i].bin[j].right, sizeof(uint32_t));
		}
	}
	outfile.close();
}
