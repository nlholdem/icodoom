#include "deep_feedback_learning.h"
#include <Iir.h>
#include <stdio.h>
#include <math.h>
#include "opencv2/core/core.hpp"

using namespace cv;

int colourByMax(double* in, int numCategories);
void printErrors(DeepFeedbackLearning* net);
void printWeights(DeepFeedbackLearning* net);



void test_spiral() {

	int nHidden[] = {50};
	const int nInputs = 2;
	const int nOutputs = 3;
	const int numDatapts = 100;
	const int numExamples = numDatapts*nOutputs;


	double input[2];
	double error[3];
	int epoch = 20;

	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(nInputs,nHidden,1,nOutputs);
	deep_fbl->initWeights(1.0,1);
	deep_fbl->setLearningRate(0.05);
	deep_fbl->setUseDerivative(0);
	deep_fbl->setBias(1.0);
	// creat the data
	double (*in)[nInputs] = new double[numExamples][nInputs];
	double (*err)[nOutputs] = new double[numExamples][nOutputs];
	double (*yhat)[nOutputs] = new double[numExamples][nOutputs];
	double (*exp_scores)[nOutputs] = new double[numExamples][nOutputs];
	double (*probs)[nOutputs] = new double[numExamples][nOutputs];
	double *correct_logProbs = new double[numExamples];
	int *out = new int[numExamples];

	char c;
	int cat;


	double r;
	double t;
	//fprintf(stdout, "Creating dataset...\n");
	// make the data
	for(int i=0; i<nOutputs; i++) {
		r = 0.0;
		for(int j=numDatapts*i; j<numDatapts*(i+1); j++) {
			cat = (int)((double)3*(((double)random())/((double)RAND_MAX)));
			t = (double)i * 4.0 + (double)j * 4 / numDatapts;
			r += 1 / (double)numDatapts;
			in[j][0] = r * sin(t);
			in[j][1] = r * cos(t);
			out[j] = i;
			fprintf(stderr, "r: %f t: %f x1: %f x2: %f class: %d\n", r, t, in[j][0], in[j][1], out[j]);
		}
	}

	for(int i=0; i< epoch; i++) {
		double loss = 0.0;
		if(i%100 == 0) {
			deep_fbl->setLearningRate(0.0);
		}
		else {
			deep_fbl->setLearningRate(0.05);
		}
		for(int j=0; j<numExamples; j++) {
			cat = (int)((double)numExamples*(((double)random())/((double)RAND_MAX)));
			
			
			//cat = (int)((double)numExamples*(((double)random())/((double)RAND_MAX)));
			//printf("j: %d\n", j);
			for(int k=0; k<nOutputs; k++) {
				err[j][k] = 0.0;
			}
			deep_fbl->doStep(in[j], err[j]);
			double sum_scores = 0.0;
			for(int k=0; k<nOutputs; k++) {
				yhat[j][k] = deep_fbl->getOutput(k);
				exp_scores[j][k] = exp(yhat[j][k]);
				sum_scores += exp_scores[j][k];
			}
			for(int k=0; k<nOutputs; k++) {
				probs[j][k] = exp_scores[j][k] / sum_scores;
				err[j][k] = - probs[j][k];
				err[j][k] = - yhat[j][k] - 1.0;
			}
			err[j][out[j]] += 2.0;
			correct_logProbs[j] = -log(probs[j][out[j]]);

			if(i%100==0) {
				
			fprintf(stdout, "target: %d", out[j]);
			for(int k=0; k<nOutputs; k++) {
				fprintf(stdout, " %d: in %f %f out %f prob %f err %f", k, in[j][0], in[j][1], yhat[j][k], probs[j][k], err[j][k]);
			}
			fprintf(stdout, "\n");
			}

			deep_fbl->doStep(in[j], err[j]);
			correct_logProbs[j] = -log(probs[j][out[j]]);
			loss += correct_logProbs[j] / numExamples;



		}
		fprintf(stdout, "iter %d loss: %f\n", i, loss);

	}
	//test the thing
	deep_fbl->setLearningRate(0.0);
	double x_pos, y_pos;
	FILE* f=fopen("test_basic.dat","wt");
	int test_output[100][100];
	for(int i=0; i<100; i++) {
		for (int j=0; j<100; j++){
			x_pos = -1.0 + (double)i*0.02;
			y_pos = -1.0 + (double)j*0.02;
			in[0][0] = x_pos;
			in[0][1] = y_pos;
			deep_fbl->doStep(in[0], err[j]);
			double sum_scores = 0.0;
			for(int k=0; k<nOutputs; k++) {
				yhat[j][k] = deep_fbl->getOutput(k);
				exp_scores[j][k] = exp(yhat[j][k]);
				sum_scores += exp_scores[j][k];
			}
			printf("in: %f %f %f %f %f\n", in[0][0], in[0][1], yhat[j][0], yhat[j][3], yhat[j][2]);

			for(int k=0; k<nOutputs; k++) {
				probs[j][k] = exp_scores[j][k] / sum_scores;
			}
			test_output[i][j] = colourByMax(probs[j], nOutputs);
			fprintf(f, "%d ", test_output[i][j]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

int colourByMax(double* in, int numCategories) {
	int colour = 0;
	double max = 0.0;
	for(int i=0; i<numCategories; i++) {
		if(in[i]>max) {
			max = in[i];
			colour += 255/numCategories;
		}
	}
	return colour;	
}

void test_basic() {
	int nHidden[] = {20,20};
	const int nInputs = 2;
	const int nOutputs = 1;
	const int numExamples = 4;


	double input[2];
	double error[1];
	double yhat;
	int epoch = 500000;
	char c;

	DeepFeedbackLearning* net = new DeepFeedbackLearning(nInputs,nHidden,2,nOutputs);
	//	net->initWeights(1.0,1);
	//	void initWeights(double max = 0.001, int initBias = 1, Neuron::WeightInitMethod weightInitMethod = Neuron::MAX_OUTPUT_RANDOM);
	net->initWeights(1.0, 1);//, Neuron::MAX_OUTPUT_CONST);
	net->setLearningRate(0.1);
	net->setUseDerivative(0);
	net->setBias(1.0);
	net->setAlgorithm(DeepFeedbackLearning::backprop);

	double inputs[4][2] = {
			{0,0},
			{0,1},
			{1,0},
			{1,1}
	};
	double targets[4] = {
			-0.9,
			0.9,
			0.9,
			-0.9
	};

	
	for (int i=0; i<epoch; i++) {
		int j = (int)((double)4*(((double)random())/((double)RAND_MAX)));

//		char c = scanf("%c", &c);
		error[0] = 0.0;
		net->doStep(inputs[j], error);
//		printWeights(net);
		yhat = net->getOutput(0);
		error[0] = targets[j] - yhat;
		net->doStep(inputs[j], error);
		printf("%d err %f\n", i, fabs(error[0]));
/*		printf("Inputs: %f %f Target: %f Output %f Error %f\n", inputs[j][0], inputs[j][1], targets[j], yhat, error[0]);
		printErrors(net);
		printWeights(net);
		printf("\n***\n");
*/
	}	
}


void test_XOR() {
	int nHidden[] = {2};
	const int nInputs = 2;
	const int nOutputs = 1;
	const int numExamples = 4;


	double input[2];
	double target[1];
	double error[1];
	double yhat;
	int epoch = 100000;
	char c;

	DeepFeedbackLearning* net = new DeepFeedbackLearning(nInputs,nHidden,1,nOutputs);
//	net->initWeights(1.0,1);
//	void initWeights(double max = 0.001, int initBias = 1, Neuron::WeightInitMethod weightInitMethod = Neuron::MAX_OUTPUT_RANDOM);
	net->initWeights(0.1, 1, Neuron::MAX_OUTPUT_RANDOM);
	net->setLearningRate(0.0);
	net->setBias(1.0);
	net->setMomentum(0.9);
	net->setUseDerivative(0);
	net->setAlgorithm(DeepFeedbackLearning::backprop);


	double inputs[4][2] = {
			{0,0},
			{0,1},
			{1,0},
			{1,1}
			};
	double targets[4] = {
			-0.8,
			0.8,
			0.8,
			-0.8
	};

//	printWeights(net);
	
	for (int i=0; i<epoch; i++) {
		int j = (int)((double)4*(((double)random())/((double)RAND_MAX)));

		
//		for (int j=0; j<4; j++) {
			fprintf(stdout, "in: %f %f out: %f\n", inputs[j][0], inputs[j][1], targets[j]);
			error[0] = 0.0;
			net->setLearningRate(0.0);
			net->doStep(inputs[j], error);
			yhat = net->getOutput(0);
			error[0] = targets[j] - yhat;
			net->setLearningRate(0.01);
			printWeights(net);
			printf("\n");
			net->doStep(inputs[j], error);
			printf("Inputs: %f %f Target: %f Output %f\n", inputs[j][0], inputs[j][1], targets[j], yhat);
			printErrors(net);
			printf("\n");
			printWeights(net);
			printf("***\n");
			
			printf("Input: %f %f Target: %f\n", inputs[j][0], inputs[j][1], targets[j]);
			
//			printf("LOSS %f\n", fabs(net->getOutput(0) - targets[j]));
			int x=scanf("%c", &c);
		
	}
	net->saveModel("testXOR.txt");
}

void printErrors(DeepFeedbackLearning* net) {
	
	for (int i=net->getNumHidLayers(); i>=0; i--) {
		Layer* layer = net->getLayer(i);
		printf("Layer %d nNeurons %d", i, layer->getNneurons());
		
		for (int j=0; j<layer->getNneurons(); j++) {
			Neuron* neuron = layer->getNeuron(j);
			if(neuron == NULL || neuron == 0) {
				printf("NULL PTR!\n");
			}
			else {
				printf(" n%d out %f error %f", j, neuron->getOutput(), neuron->getError());
			}
		}
		printf("\n");
	}
}

void printWeights(DeepFeedbackLearning* net) {
	for (int i=net->getNumHidLayers(); i>=0; i--) {
		Layer* layer = net->getLayer(i);
		printf("Layer %d has %d neurons \n", i, layer->getNneurons());
		
		for (int j=0; j<layer->getNneurons(); j++) {
			Neuron* neuron = layer->getNeuron(j);
			if(neuron == NULL || neuron == 0) {
				printf("NULL PTR!\n");
			}
			else {
				printf(" N %d", j);
				for (int k=0; k<neuron->getNinputs(); k++) {
					printf(" w%d %f ch %f ", k, neuron->getAvgWeight(k), neuron->getAvgWeightCh(k));
				}
			}
			printf("\n");
		}
	}
	
}


void testXOR_ICO() {
	int nHidden[] = {2};
	const int nInputs = 2;
	const int nOutputs = 1;
	const int numExamples = 4;


	double input[2];
	double target[1];
	double error[2];
	double yhat;
	int epoch = 100000;
	char c;

	DeepFeedbackLearning* net = new DeepFeedbackLearning(nInputs,nHidden,1,nOutputs);
//	net->initWeights(1.0,1);
//	void initWeights(double max = 0.001, int initBias = 1, Neuron::WeightInitMethod weightInitMethod = Neuron::MAX_OUTPUT_RANDOM);
	net->initWeights(1.0, 1, Neuron::MAX_OUTPUT_CONST);
	net->setLearningRate(0.0);
	net->setMomentum(0.9);
	net->setUseDerivative(0);
	net->setBias(1.0);
	net->setAlgorithm(DeepFeedbackLearning::ico);

	printf("Num Layers: %d\n", net->getNumHidLayers()+1);
	Layer *layer;
	layer = net->getLayer(0);
	layer = net->getLayer(1);
//	layer = net->getLayer(20);
//	assert(1==0);
	
	printf("Save success: %d\n", net->saveModel("testXOR_ICO.txt"));
	printf("Load success: %d\n", net->loadModel());

//	return;


	double inputs[4][2] = {
			{0,0},
			{0,1},
			{1,0},
			{1,1}
			};
	double targets[4] = {
			-0.9,
			0.9,
			0.9,
			-0.9
	};

	printWeights(net);

	for (int i=0; i<epoch; i++) {
		int j = (int)((double)4*(((double)random())/((double)RAND_MAX)));


//		for (int j=0; j<4; j++) {
//			fprintf(stdout, "in: %f %f out: %f\n", inputs[j][0], inputs[j][1], targets[j]);
			error[0] = 0.0;
			net->setLearningRate(0.0);
			net->doStep(inputs[j], error);
			yhat = net->getOutput(0);
			error[0] = targets[j] - yhat;
			error[1] = targets[j] - yhat;

			net->setLearningRate(0.05);
//			printWeights(net);
//			printf("\n");
			net->doStep(inputs[j], error);
//			printf("Inputs: %f %f Target: %f Output %f\n", inputs[j][0], inputs[j][1], targets[j], yhat);
//			printErrors(net);
//			printf("\n");
//			printWeights(net);
//			printf("***\n");
//			int x=scanf("%c", &c);


//			printf("Input: %f %f Target: %f\n", inputs[j][0], inputs[j][1], targets[j]);

			printf("LOSS %f\n", fabs(net->getOutput(0) - targets[j]));
			int x=scanf("%c", &c);

	}
}


int main(int, char**) {
	//test_basic();
	test_XOR();
	//testXOR_ICO();
}
