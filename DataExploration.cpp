/*
Module Name : Data Exploration
Date : 2023 - 2 - 4
Author : Naomi Zilber

Module Purpose
Recreate statistical function from R in C++

Module Design Description
Read in a file and code the functions in C++ that will be applied to the data read in

Inputs:
Boston.csv file

Outputs:
Display the results of calling the statistical functions
*/

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

// Find the sum of a numeric vector
double sum(vector<double> v) {
    double sum = 0;
    // add every value in v to the sum
    for (double n : v) {
        sum += n;
    }

    return sum;
}

// Find the mean of a numeric vector
double mean(vector<double> v) {
    return sum(v) / v.size();
}

// Find the median of a numeric vector
double median(vector<double> v) {
    // sort the vector
    sort(v.begin(), v.end());

    // if the vector is of even length then median=average of middle two values
    if (v.size() % 2 == 0) {
        double median = (v[v.size() / 2] + v[(v.size() / 2) - 1]) / 2;
        return median;
    }
    else {
        return v[v.size() / 2];
    }
}

// Find the range of a numeric vector
double range(vector<double> v) {
    // sort the vector to be able to find the max and min values to compute the range
    sort(v.begin(), v.end());
    return v[v.size() - 1] - v[0];
}

// Compute the covarriance between two numeric vectors
double covar(vector<double> rm, vector<double> medv) {
    double sum = 0;
    double rm_mean = mean(rm);
    double medv_mean = mean(medv);

    for (int i = 0; i < rm.size(); i++) {
        sum += (rm[i] - rm_mean) * (medv[i] - medv_mean);
    }

    return sum / (rm.size() - 1);
}

// Compute the correlation between two numeric vectors
double cor(vector<double> rm, vector<double> medv) {
    double sum = 0;
    double rm_mean = mean(rm);
    double medv_mean = mean(medv);

    // compute a part of the variance of rm
    for (double num : rm) {
        sum += (num - rm_mean) * (num - rm_mean);
    }
    // can now find sigma by taking the square root of the variance
    double rm_sigma = sqrt(sum / (rm.size() - 1));

    sum = 0;
    for (double num : medv) {
        sum += (num - medv_mean) * (num - medv_mean);
    }
    double medv_sigma = sqrt(sum / (medv.size() - 1));

    // find the product of the sigmas
    double sigma_prod = rm_sigma * medv_sigma;

    return covar(rm, medv) / sigma_prod;
}

// Print out the basic stats about the vector, including its sum, mean, median, and range
void print_stats(vector<double> v) {
    cout << "Sum = " << sum(v) << endl;
    cout << "Mean = " << mean(v) << endl;
    cout << "Median = " << median(v) << endl;
    cout << "Range = " << range(v) << endl;
}

int main(int argc, char** argv) {
    ifstream inFS;
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    // attempt to open the file
    cout << "Opening file Boston.csv." << endl;

    inFS.open("Boston.csv");
    if (!inFS.is_open()) {
        cout << "Could not open file Boston.csv." << endl;
        return 1;   // 1=error
    }

    // can now use inFS stream like cin stream
    // file Boston.csv should contain two doubles

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // echo heading
    cout << "heading: " << line << endl;

    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length " << rm.size() << endl;
    cout << "Closing file Boston.csv." << endl;
    inFS.close();

    cout << "Number of records: " << numObservations << endl;
    cout << "\nStats for rm" << endl;
    print_stats(rm);

    cout << "\nStats for medv" << endl;
    print_stats(medv);

    cout << "\nCovariance = " << covar(rm, medv) << endl;
    cout << "\nCorrelation = " << cor(rm, medv) << endl;
    cout << "\nProgram terminated" << endl;

    return 0;
}