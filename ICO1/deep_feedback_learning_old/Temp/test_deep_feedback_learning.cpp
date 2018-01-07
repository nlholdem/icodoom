#include "deep_feedback_learning.h"
#include <Iir.h>
#include<stdio.h>

void printErrors(FILE *f, DeepFeedbackLearning* net);
void printWeights(FILE *f, DeepFeedbackLearning* net);


// inits the network with random weights with quite a few hidden units so that
// a nice response is generated
void test_forward() {
	int nFiltersInput = 10;
	int nFiltersHidden = 10;
	int nHidden[] = {10,10};

	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,2,1,nFiltersInput,nFiltersHidden,100,200);
	FILE* f=fopen("test_deep_fbl_cpp_forward.dat","wt");
	deep_fbl->setLearningRate(0.0);
	// random init
	deep_fbl->initWeights(0.1);

	double input[2];
	double error[2];

	for(int n = 0; n < 1000;n++) {

		input[0] = 0;
		if ((n>10)&&(n<20)) {
			input[0] = 0.1;
		}
		fprintf(f,"%f ",input[0]);

		deep_fbl->doStep(input,error);
		for(int i=0; i<deep_fbl->getNumHidLayers(); i++) {
			fprintf(f,"%f ",deep_fbl->getLayer(i)->getNeuron(0)->getSum());
		}
		fprintf(f,"%f ",deep_fbl->getOutputLayer()->getNeuron(0)->getOutput());
		
		fprintf(f,"\n");
		
	}

	fclose(f);
}



void test_learning() {
	int nHidden[] = {2};
	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,1,1);
	deep_fbl->initWeights(0.1,0,Neuron::MAX_OUTPUT_CONST);
	deep_fbl->setLearningRate(0.01);
	deep_fbl->setAlgorithm(DeepFeedbackLearning::ico);
	
	FILE* f=fopen("test_deep_fbl_cpp_learning.dat","wt");

	double input[2];
	double error[2];	
	
	for(int n = 0; n < 1000;n++) {
		
		double stim = 0;
		double err = 0;
		if ((n>10)&&(n<1000)) {
			stim = 1;
			if ((n>500)&&(n<600)) {
				err = 1;
			}
			if ((n>700)&&(n<800)) {
				err = -1;
			}
		}
		fprintf(f,"%f %f ",stim,err);

		input[0] = stim;
		error[0] = err;
		error[1] = err;

		deep_fbl->doStep(input,error);

		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				for(int k=0; k<deep_fbl->getNumHidLayers(); k++) {
					fprintf(f, "%f ",
							deep_fbl->getLayer(k)->getNeuron(i)->getWeight(j));
				}
			}
		}
		for(int i=0;i<1;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%f ",
					deep_fbl->getOutputLayer()->getNeuron(i)->getWeight(j));
			}
		}
		for(int i=0;i<1;i++) {
			fprintf(f,
				"%f ",
				deep_fbl->getOutputLayer()->getNeuron(i)->getOutput());
		}
		fprintf(f,"\n");
		
	}

	fclose(f);
}

//create of version of XOR for testing DFL vs feedback-error learning
void test_feedback_learning() {
	int nHiddenLayers = 1;
	int nHidden[] = {2};
	int nFiltersInput = 0;
	int nFiltersHidden = 0;
	double minT = 3;
	double maxT = 15;

	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,nHiddenLayers,1,nFiltersInput, nFiltersHidden, minT,maxT);
	deep_fbl->initWeights(0.3,1.0,Neuron::MAX_OUTPUT_RANDOM);
	deep_fbl->setLearningRate(0.01);
	deep_fbl->setMomentum(0.0);
	deep_fbl->setAlgorithm(DeepFeedbackLearning::ico);
	deep_fbl->setUseDerivative(0);
	deep_fbl->setBias(1.0);

	FILE* f=fopen("test_deep_fbl_cpp_feedback_learning.dat","wt");
	FILE* fw=fopen("test_deep_fbl_cpp_feedback_learning.wts","wt");
	FILE* fe=fopen("test_deep_fbl_cpp_feedback_learning.err","wt");

	double input[2];
	double error[2];
	double state;
	double reflex;
	double gain = 0.1;
	double netgain = 1.0;

	double inputs[4][2] = {
			{0,0},
			{1,1},
			{0,0},
			{1,1}
			};
	double targets[4] = {
			0.0,
			0.9,
			0.0,
			0.9
	};
	int indx;

	int rep = 200;
	int epoch=1;
	int term = 100000;

	int x;
	char c;

	for (int e=0; e<epoch; e++) {
		state=0.0;
		for(int n = 0; n < term;n++) {

			double stim = 0;
			double err = 0;

			input[0] = input[1] = 0.0;
			if (((n%rep)==0)) {
				indx = (int)((double)4*(((double)random())/((double)RAND_MAX)));
			}

			if (((n%rep)==100)) {
				state += targets[indx];
			}

/*			if ((n%rep)==101) {
				input[0] = inputs[indx][0];
				input[1] = inputs[indx][1];
			}
*/
			// scale the inputs with the magnitude of the deviation - otherwise we're asking it to learn a one->many mapping!
			if (((n%rep)==80)) {
				input[0] = inputs[indx][0];
				input[1] = inputs[indx][1];
			}

			error[0] = 0.0;
			error[1] = 0.0;
			deep_fbl->setLearningRate(0.0);
			deep_fbl->doStep(input,error);
			// give the FF controller a chance to neutralise the error before calculating the reflex
			state += netgain*(deep_fbl->getOutput(0));
			reflex = -(gain * state);
			error[0] = reflex;
			error[1] = reflex;
			deep_fbl->setLearningRate(0.0);
			deep_fbl->doStep(input,error);
			state += reflex;

			fprintf(f,"%e %e %e %e %e %d ", input[0], input[1],
										state, reflex, targets[indx], indx);


			for(int i=0; i<deep_fbl->getNumHidLayers()+1; i++) {
				for(int j=0; j<deep_fbl->getLayer(i)->getNneurons(); j++) {
					fprintf(f, "%e ", deep_fbl->getLayer(i)->getNeuron(j)->getError());
				}
			}
/*
			for(int i=0;i<2;i++) {
				for(int j=0;j<2;j++) {
					for(int k=0; k<deep_fbl->getNumHidLayers(); k++) {
						fprintf(f,
								"%e ",
								deep_fbl->getLayer(k)->getNeuron(i)->getWeight(j));
					}
				}
			}
			for(int i=0;i<1;i++) {
				for(int j=0;j<2;j++) {
					fprintf(f,
						"%e ",
						deep_fbl->getOutputLayer()->getNeuron(i)->getWeight(j));
				}
			}
			*/
			for(int i=0;i<1;i++) {
				fprintf(f,
					"%e ",
					deep_fbl->getOutputLayer()->getNeuron(i)->getOutput());
			}

			fprintf(f,"\n");
			fflush(f);
			printWeights(fw, deep_fbl);
			printErrors(fe, deep_fbl);
//			x=scanf("%c", &c);
//			printf("Step %d\n", n);




		}
	}
	fclose(f);
	deep_fbl->saveModel("testDFL.txt");
}

void test_learning_and_filters() {
	int nHidden[] = {2};
	int nFiltersInput = 5;
	int nFiltersHidden = 5;
	double minT = 3;
	double maxT = 15;
	
	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,1,1,nFiltersInput, nFiltersHidden, minT,maxT);
	deep_fbl->initWeights(0.001,0,Neuron::MAX_OUTPUT_CONST);
	deep_fbl->setLearningRate(0.0001);
	deep_fbl->setAlgorithm(DeepFeedbackLearning::ico);
	deep_fbl->setBias(0);
	
	FILE* f=fopen("test_deep_fbl_cpp_learning_with_filters.dat","wt");

	double input[1];
	double error[2];

	int rep = 200;
	
	for(int n = 0; n < 10000;n++) {
		
		double stim = 0;
		double err = 0;
		if (((n%rep)>100)&&((n%rep)<105)) {
			stim = 1;
		}
		if (((n%rep)>105)&&((n%rep)<110)&&(n<9000)) {
			err = 1;
		}
		fprintf(f,"%e %e ",stim,err);

		input[0] = stim;
		error[0] = err;
		error[1] = err;

		deep_fbl->doStep(input,error);

		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				for(int k=0; k<deep_fbl->getNumHidLayers(); k++) {
					fprintf(f,
							"%e ",
							deep_fbl->getLayer(k)->getNeuron(i)->getWeight(j));
				}
			}
		}
		for(int i=0;i<1;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%e ",
					deep_fbl->getOutputLayer()->getNeuron(i)->getWeight(j));
			}
		}
		for(int i=0;i<1;i++) {
			fprintf(f,
				"%e ",
				deep_fbl->getOutputLayer()->getNeuron(i)->getOutput());
		}
		fprintf(f,"\n");
		
	}

	fclose(f);
}

void printErrors(FILE *fe, DeepFeedbackLearning* net) {

	for (int i=net->getNumHidLayers(); i>=0; i--) {
		Layer* layer = net->getLayer(i);

		for (int j=0; j<layer->getNneurons(); j++) {
			Neuron* neuron = layer->getNeuron(j);
			if(neuron == NULL || neuron == 0) {
				printf("NULL PTR!\n");
			}
			else {
				fprintf(fe, " n%d out %f error %f", j, neuron->getOutput(), neuron->getError());
			}
		}
	}
	fprintf(fe, "\n");
	fflush(fe);
}

void printWeights(FILE *fw, DeepFeedbackLearning* net) {
	for (int i=net->getNumHidLayers(); i>=0; i--) {
		Layer* layer = net->getLayer(i);

		for (int j=0; j<layer->getNneurons(); j++) {
			Neuron* neuron = layer->getNeuron(j);
			if(neuron == NULL || neuron == 0) {
				printf("NULL PTR!\n");
			}
			else {
				for (int k=0; k<neuron->getNinputs(); k++) {
					for (int l=0; l<neuron->getNfilters(); l++) {
						fprintf(fw, " w[%d][%d][%d][%d] %f ch %f fIn %f", i, j, k, l, neuron->getWeight(k,l), neuron->getWeightChange(k,l), neuron->getFilteredInput(k,l));

					}
				}
			}
		}
	}
	fprintf(fw, "\n");
	fflush(fw);
}




int main(int,char**) {
	test_forward();
	test_learning();
	test_learning_and_filters();
	test_feedback_learning();
}
