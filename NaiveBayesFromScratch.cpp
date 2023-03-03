/*
Module Name : Naive Bayes from Scratch
Date : 2023 - 3 - 4
Author : Naomi Zilber

Module Purpose
Write a program to perform naive Bayes from scratch in C++

Module Design Description
Read in a file and code the functions in C++ that will recreate a naive Bayes model

Inputs:
titanic_project.csv file

Outputs:
Display the probabilties, test metrics, and run time of the naive Bayes model
*/

#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

/* find the accuracy of the model using the formula accuracy = correct/total
 * predicted = vector of probabilities 1 or 0
 * lbls = survived vector
 */
double accuracy(vector<double> predicted, vector<double> lbls) {
    double correct = 0;

    // find the amount of correct predictions
    for (int i = 0; i < predicted.size(); i++) {
        if (predicted[i] == lbls[i]) {
            correct++;
        }
    }

    // calculate accuracy
    double acc = correct / predicted.size();

    return acc;
}

/* find the sensitivity of the model using the formula sensitivity = tp/(tp+fn)
 * predicted = vector of probabilities 1 or 0
 * lbls = survived vector
 */
double sensitivity(vector<double> predicted, vector<double> lbls) {
    vector<double> predictions = predicted;
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
 * predicted = vector of probabilities 1 or 0
 * lbls = survived vector
 */
double specificity(vector<double> predicted, vector<double> lbls) {
    vector<double> predictions = predicted;
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

// find the mean value of the given vector
double mean(vector<double> v1) {
    double sum = 0;

    // find the sum of all values in the vector
    for (double n : v1) {
        sum += n;
    }

    // calculate and return the mean
    return sum / v1.size();
}

// calculate the variance of the given vector
double variance(vector<double> v1) {
    double v1_mean = mean(v1);
    double sum = 0;

    for (double n : v1) {
        sum += (n - v1_mean) * (n - v1_mean);
    }

    double var = sum / (v1.size() - 1);
    return var;
}

// calculate likelihood of age (a quantitative variable) using a formula
double calcAgeLikelihood(double v1, double v1_mean, double v1_var) {
    double lh_age = 0;

    double part1 = 1 / sqrt(2 * M_PI * v1_var);
    double squared = (v1 - v1_mean) * (v1 - v1_mean) * -1;
    lh_age = part1 * exp(squared / (2 * v1_var));

    return lh_age;
}

/* takes in two vectors and their target values
 * counts the amount of observations that have both of the target values in their
 * respective vectors
*/
double getLength(vector<double> v1, double v1_num, vector<double> v2, double v2_num) {
    int count = 0;
    for (int i = 0; i < v1.size(); i++) {
        if (v1_num == v1[i] && v2_num == v2[i]) {
            count++;
        }
    }

    return count;
}

// find the amount of people that survived and perished
// returns a vector containing {perished_count, survived_count}
vector<double> getSurvivedCounts(vector<double> survived) {
    double yes = 0;
    double no = 0;

    // find the survived and perished counts
    for (double n : survived) {
        if (n == 1) {
            yes++;
        }
        else {
            no++;
        }
    }

    return { no, yes };
}

// calculates the a-priori of the vector survived
vector<double> getApriori(vector<double> survived) {
    // gets the survived and perished counts
    vector<double> counts = getSurvivedCounts(survived);

    // calculates and returns a-priori
    return { counts[0] / survived.size(), counts[1] / survived.size() };
}

// calculates the sex likelihood values; P(sex|survived)
vector<vector<double>> sexLikelihood(vector<double> survived, vector<double> sex) {
    // get counts for survived {no, yes}
    vector<double> counts = getSurvivedCounts(survived);

    // likelihood for sex
    vector<vector<double>> lh_sex(2, vector<double>(2));
    // for every survived or perished
    for (int sv : {0, 1}) {
        // for every sex (female and male)
        for (int sx : {0, 1}) {
            // calculate likelihood
            lh_sex[sx][sv] = getLength(sex, sx, survived, sv) / counts[sv];
        }
    }

    return lh_sex;
}

// calculates the pclass likelihood values; P(pclass|survived)
vector<vector<double>> pclassLikelihood(vector<double> survived, vector<double> pclass) {
    // get counts for survived {no, yes}
    vector<double> counts = getSurvivedCounts(survived);

    // likelihood for pclass
    vector<vector<double>> lh_pclass(3, vector<double>(2));
    // for every survived or perished
    for (int sv : {0, 1}) {
        // for every class
        for (int pc : {1, 2, 3}) {
            // get likelihood
            lh_pclass[pc - 1][sv] = getLength(pclass, pc, survived, sv) / counts[sv];
        }
    }

    return lh_pclass;
}

// computes the mean and variance of the vector age since it is a quantitative variable
// so these are needed to calculate its likelihood
vector<vector<double>> likelihoodQuan(vector<double> survived, vector<double> age) {
    vector<double> age_mean = { 0,0 };
    vector<double> age_variance = { 0,0 };

    // for every survived or perished
    for (int sv : {0, 1}) {
        vector<double> temp(age.size());
        int j = 0;
        // fill temp with the ages of only the people who survived (when sv=1)/perished (when sv=0)
        for (int i = 0; i < survived.size(); i++) {
            if (survived[i] == sv) {
                temp[j] = age[i];
                j++;
            }
        }

        temp.resize(j);

        // get the mean and variance for the people who survived/perished
        age_mean[sv] = mean(temp);
        age_variance[sv] = variance(temp);
    }

    return {age_mean, age_variance};
}

// calculate raw probabilities
// calculates the a-priori and likelihoods again
vector<vector<double>> calcRawProb(vector<vector<double>> train_data, vector<vector<double>> test_data) {    
    // make naive Bayes model
    // compute the a-priori, pclass likelihood, sex likelihood, and age means and variances
    vector<double> apriori = getApriori(train_data[3]);
    vector<vector<double>> lh_pclass = pclassLikelihood(train_data[3], train_data[0]);
    vector<vector<double>> lh_sex = sexLikelihood(train_data[3], train_data[1]);
    vector<vector<double>> age_metrics = likelihoodQuan(train_data[3], train_data[2]);

    // predicted probabilities for surviving and perishing for every observation
    vector<vector<double>> predicted(test_data[0].size());

    for (int i = 0; i < predicted.size(); i++) {
        // for every observation
        double pc = test_data[0][i];
        double sx = test_data[1][i];
        double age = test_data[2][i];

        // compute likelihood times prior for survival
        double num_s = lh_pclass[pc - 1][1] * lh_sex[sx][1] * apriori[1] *
            calcAgeLikelihood(age, age_metrics[0][1], age_metrics[1][1]);

        // compute likelihood times prior for perishing
        double num_p = lh_pclass[pc - 1][0] * lh_sex[sx][0] * apriori[0] *
            calcAgeLikelihood(age, age_metrics[0][0], age_metrics[1][0]);

        double denominator = num_s + num_p;

        // calculate the probabilities of a person surviving or perishing for the observation
        double prob_survived = num_s / denominator;
        double prob_perished = num_p / denominator;

        // set the prediction for that observation to the probabilities of perishing and surviving
        predicted[i] = { prob_perished, prob_survived };
    }

    return predicted;
}

// round the predicted probabilities to 1 or 0
vector<double> roundProbs(vector<vector<double>> predicted) {
    vector<double> probs(predicted.size());

    // if a survived probability > 0.5 then set probability to 1; otherwise, set it to 0
    for (int i = 0; i < predicted.size(); i++) {
        if (predicted[i][1] > 0.5) {
            probs[i] = 1;
        }
        else {
            probs[i] = 0;
        }
    }

    return probs;
}

// print out the values in the given matrix v
void printProbs(vector<vector<double>> v) {
    for (int i = 0; i < v[0].size(); i++) {
        for (int j = 0; j < v.size(); j++) {
            cout << v[j][i] << " ";
        }
        cout << endl;
    }
    cout << endl;
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
    vector<vector<double>> train(4, vector<double>(800));
    for (int i = 0; i < 800; i++) {
        train[0][i] = pclass[i];
        train[1][i] = sex[i];
        train[2][i] = age[i];
        train[3][i] = survived[i];
    }

    // test data
    int j = 0;
    vector<vector<double>> test(4, vector<double>(survived.size()-800));
    for (int i = 800; i < survived.size(); i++) {
        test[0][j] = pclass[i];
        test[1][j] = sex[i];
        test[2][j] = age[i];
        test[3][j] = survived[i];
        j++;
    }

    time_point<system_clock> start, end;
    // get the current time before the algorithm starts
    start = system_clock().now();

    // compute the naive bayes model
    // compute the a-priori, pclass likelihood, sex likelihood, and age means and variances
    vector<double> apriori = getApriori(train[3]);
    vector<vector<double>> lh_pclass = pclassLikelihood(train[3], train[0]);
    vector<vector<double>> lh_sex = sexLikelihood(train[3], train[1]);
    vector<vector<double>> age_metrics = likelihoodQuan(train[3], train[2]);
    // calculate the standard deviations from the variances
    age_metrics[1][0] = sqrt(age_metrics[1][0]);
    age_metrics[1][1] = sqrt(age_metrics[1][1]);

    // get the current time when the algorithm finished
    end = system_clock().now();
    // calculate the total time the algorithm ran
    duration<double> elapsed_time = end - start;

    // output all of the probabilities of the model
    cout << "A-priori probabilities" << endl << apriori[0] << " " << apriori[1] << endl << endl;
    
    cout << "Conditional probabilities" << endl;
    cout << "pclass" << endl;
    printProbs(lh_pclass);

    cout << "sex" << endl;
    printProbs(lh_sex);

    cout << "age" << endl;
    printProbs(age_metrics);

    // compute predicted values from test data
    vector<vector<double>> predicted = calcRawProb(train, test);

    cout << "Metrics" << endl;
    // get the rounded probabilities
    vector<double> probs = roundProbs(predicted);
    // get accuracy and output it
    double acc = accuracy(probs, test[3]);
    cout << "accuracy = " << acc << endl;

    // get sensitivity and specificity
    double sensitive = sensitivity(probs, test[3]);
    double spec = specificity(probs, test[3]);

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