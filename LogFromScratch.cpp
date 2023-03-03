/*
Module Name : Logistic Regression from Scratch
Date : 2023 - 3 - 4
Author : Naomi Zilber

Module Purpose
Write a program to perform logistic regression from scratch in C++

Module Design Description
Read in a file and code the functions in C++ that will recreate a logistic regression model

Inputs:
titanic_project.csv file

Outputs:
Display the coefficients, test metrics, and run time of the logistic regression model
The weights w0 and w1 represent the intercept and sexmale coefficients when ran in R Studio, respectively
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;

// sigmoid function
double sigmoid(double z) {
    return 1.0 / (1 + exp(-1 * z));
}

// computes the sigmoid values for every observation in the input matrix
vector<double> findSigValues(vector<vector<double>> matrix, vector<double> weights) {
    vector<double> sigValues(matrix[0].size());

    // generate sigmoid vector
    for (int i = 0; i < matrix[0].size(); i++) {
        double z = 0;
        // multiply every row by the weights
        for (int j = 0; j < matrix.size(); j++) {
            z += matrix[j][i] * weights[j];
        }

        // find the sigmoid value
        sigValues[i] = sigmoid(z);
    }

    return sigValues;
}

// subtracts elements of v2 from elements of v1 and returns resulting vector
vector<double> vectorSubtraction(vector<double> v1, vector<double> v2) {
    vector<double> result(v1.size());

    // for every element, subtract v2[i] from v1[i] and set result[i] equal to that
    for (int i = 0; i < v1.size(); i++) {
        result[i] = v1[i] - v2[i];
    }

    return result;
}

// adds elements of v2 to elements of v1 and returns resulting vector
vector<double> vectorAddition(vector<double> v1, vector<double> v2) {
    vector<double> result(v1.size());

    // for every index i, add the elements in that position in v1 and v2 and set result[i] equal to that
    for (int i = 0; i < v1.size(); i++) {
        result[i] = v1[i] + v2[i];
    }

    return result;
}

// divides elements of v1 by elements of v2 and returns resulting vector
vector<double> vectorDivision(vector<double> v1, vector<double> v2) {
    vector<double> result(v1.size());

    // for all elements, divided v1[i] by v2[i] and set result[i] equal to that
    for (int i = 0; i < v1.size(); i++) {
        result[i] = v1[i] / v2[i];
    }

    return result;
}

// take the power of eof every element in the vector and return it as a new vector
// so in the resulting vector every index has result[i] = (e^v1[i])
vector<double> vectorExp(vector<double> v1) {
    vector<double> result(v1.size());

    for (int i = 0; i < v1.size(); i++) {
        result[i] = exp(v1[i]);
    }

    return result;
}

// finds and returns the transpose of a matrix
vector<vector<double>> matrixTranspose(vector<vector<double>> matrix) {
    // result is the transpose matrix
    vector<vector<double>> result(matrix[0].size(), vector<double>(matrix.size()));

    // set all rows of input matrix to columns of the result matrix
    for (int i = 0; i < matrix[0].size(); i++) {
        vector<double> temp(matrix.size());
        // get the row of the input matrix
        for (int j = 0; j < matrix.size(); j++) {
            temp[j] = matrix[j][i];
        }

        // set result column to temp
        result[i] = temp;
    }

    return result;
}

// a matrix nxm is multipled by a matrix of mx1, resulting in matrix nx1
vector<double> matrixMultiplication(vector<vector<double>> matrix, vector<double> v1) {
    vector<double> result(matrix[0].size());

    for (int i = 0; i < matrix[0].size(); i++) {
        double z = 0;
        // find sum of multiplying an entire row of the input matrix by a column of v1
        for (int j = 0; j < matrix.size(); j++) {
            z += matrix[j][i] * v1[j];
        }
        // set column in matrix result to that sum (z)
        result[i] = z;
    }

    return result;
}

// multiply the matrix by a constant (scalar)
vector<double> scalarMultiplication(vector<double> matrix, double scalar) {
    vector<double> result(matrix.size());

    // multiply every element in the input matrix by the scalar
    for (int i = 0; i < matrix.size(); i++) {
        result[i] = matrix[i] * scalar;
    }

    return result;
}

// compute the predicted values
// weights = calculated coefficients; test_matrix = test data
vector<double> predictValues(vector<double> weights, vector<vector<double>> test_matrix) {
    // multiply the test matrix by the weight coefficients from the training algorithm
    vector<double> predicted = matrixMultiplication(test_matrix, weights);

    // raise every value to be e^value
    vector<double> predExp = vectorExp(predicted);
    // makes a vector that consists of only 1s
    vector<double> ones(predExp.size(), 1.0);
    // adds the vectors predExp and ones together
    vector<double> predExpPlusOne = vectorAddition(predExp, ones);
    // divides predExp by predExpPlusOne to get probabilities
    vector<double> probs = vectorDivision(predExp, predExpPlusOne);

    return probs;
}

// round the predicted probabilities to 1 or 0
// probs = predicted probabilities
vector<double> roundProbs(vector<double> probs) {
    vector<double> predictions(probs.size());

    // if a survived probability > 0.5 then set probability to 1; otherwise, set it to 0
    for (int i = 0; i < probs.size(); i++) {
        if (probs[i] > 0.5) {
            predictions[i] = 1;
        }
        else {
            predictions[i] = 0;
        }
    }
    return predictions;
}

/* find the accuracy of the model using the formula accuracy = correct/total
 * predictions = vector of probabilities 1 or 0
 * lbls = survived vector
 */
double accuracy(vector<double> predictions, vector<double> lbls) {
    double correct = 0;
    // find the amount of correct predictions
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i] == lbls[i]) {
            correct++;
        }
    }

    // calculate accuracy
    double acc = correct / predictions.size();

    return acc;
}

/* find the sensitivity of the model using the formula sensitivity = tp/(tp+fn)
 * predictions = vector of probabilities 1 or 0
 * lbls = survived vector
 */
double sensitivity(vector<double> predictions, vector<double> lbls) {
    // tp = true positive; fn = false negative
    double tp = 0;
    double fn = 0;

    // find tp and fn
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i] == 0 && lbls[i] == 0) {
            tp++;
        }
        else if (predictions[i] == 1 && lbls[i] == 0) {
            fn++;
        }
    }

    // check that the denominator isn't zero because then sensitivity is undefined
    if ((tp + fn) == 0) {
        return -1;
    }

    // calculate and return sensitivity
    return tp / (tp + fn);
}

/* find the specificity of the model using the formula specificity = tn/(tn+fp)
 * predictions = vector of probabilities 1 or 0
 * lbls = survived vector
 */
double specificity(vector<double> predictions, vector<double> lbls) {
    // tn = true negative; fp = false positive
    double tn = 0;
    double fp = 0;

    // find tn and fp
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i] == 1 && lbls[i] == 1) {
            tn++;
        }
        else if (predictions[i] == 0 && lbls[i] == 1) {
            fp++;
        }
    }

    // check that the denominator isn't zero because then specificity is undefined
    if ((tn + fp) == 0) {
        return -1;
    }

    // calculate and return specificity
    return tn / (tn + fp);
}

// Computes the coefficients of the logistic regression function
vector<double> logistic(vector<vector<double>> matrix, vector<double> lbls) {
    vector<double> weights = { 1,1 };
    double learning_rate = 0.001;
    vector<double> labels = lbls;

    vector<vector<double>> data_matrix = matrix;
    vector<double> probs(data_matrix[0].size());
    vector<double> errors(data_matrix[0].size());

    // gradient descent for 50000 iterations
    for (int i = 1; i < 50000; i++) {
        // get the sigmoid values of the input matrix
        probs = findSigValues(data_matrix, weights);
        // caculate errors
        errors = vectorSubtraction(labels, probs);
        // calculate new weights
        vector<double> temp = matrixMultiplication(matrixTranspose(data_matrix), errors);
        weights = vectorAddition(weights, scalarMultiplication(temp, learning_rate));
    }

    return weights;
}

int main(int argc, char** argv) {
    ifstream inFS;
    string line;
    string pclass_in, survived_in, sex_in, age_in;
    const int MAX_LEN = 1100;
    vector<double> pclass(MAX_LEN);
    vector<double> survived(MAX_LEN);
    vector<double> sex(MAX_LEN);
    vector<double> age(MAX_LEN);

    // attempt to open the file
    cout << "Opening file titanic_project.csv." << endl;

    inFS.open("titanic_project.csv");
    if (!inFS.is_open()) {
        cout << "Could not open file" << endl;
        return 1;   // 1=error
    }

    // can now use inFS stream like cin stream
    // file titanic_project.csv should contain 4 doubles

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // echo heading
    cout << "heading: " << line << endl;

    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, line, ',');
        // read the data into the corresponding category
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');

        // convert the data from string to integer
        pclass.at(numObservations) = stof(pclass_in);
        survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);
        age.at(numObservations) = stof(age_in);

        numObservations++;
    }

    // resize each vector to be an exact fit
    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    cout << "new length " << pclass.size() << endl;
    cout << "Closing file" << endl;
    inFS.close();

    cout << "Number of records: " << numObservations << endl << endl;

    // train data
    vector<vector<double>> train(2, vector<double>(800));
    for (int i = 0; i < 800; i++) {
        train[0][i] = survived[i];
        train[1][i] = sex[i];
    }

    // test data
    int j = 0;
    vector<vector<double>> test(2, vector<double>(survived.size()-800));
    for (int i = 800; i < survived.size(); i++) {
        test[0][j] = survived[i];
        test[1][j] = sex[i];
        j++;
    }

    // make the input data matrix for training
    vector<vector<double>> data_matrix(2, vector<double>(train[0].size()));
    for (int i = 0; i < train[0].size(); i++) {
        data_matrix[0][i] = 1;
        data_matrix[1][i] = sex[i];
    }

    // get the current time before the algorithm starts
    time_point<system_clock> start, end;
    start = system_clock().now();

    // calculate the weights (coefficients) of the logistic regression
    vector<double> weights = logistic(data_matrix, train[0]);
    // get the current time when the algorithm finished
    end = system_clock().now();
    cout << "w0 = " << weights[0] << endl << "w1 = " << weights[1] << endl << endl;

    // total time the algorithm ran
    duration<double> elapsed_time = end - start;

    // make the input data matrix for testing
    vector<vector<double>> test_matrix(2, vector<double>(test[0].size()));
    for (int i = 0; i < test[0].size(); i++) {
        test_matrix[0][i] = 1;
        test_matrix[1][i] = test[1][i];
    }

    // get the predicted values and then round the probabilities
    vector<double> predicted = predictValues(weights, test_matrix);
    vector<double> predictions = roundProbs(predicted);

    cout << "Metrics" << endl;
    // get accuracy and output it
    double acc = accuracy(predictions, test[0]);
    cout << "accuracy = " << acc << endl;

    // get sensitivity and specificity
    double sensitive = sensitivity(predictions, test[0]);
    double spec = specificity(predictions, test[0]);

    // check that sensitivity and specificity are not undefined (NA)
    if (sensitive == -1) {
        cout << "sensitivity = NA" << endl;
    }
    else {
        cout << "sensitivity = " << sensitive << endl;
    }

    if (spec == -1) {
        cout << "specificity = NA" << endl;
    }
    else {
        cout << "specificity = " << spec << endl;
    }

    // output the training time of the algorithm
    cout << "elapsed time (seconds) = " << elapsed_time.count() << endl;

    return 0;
}