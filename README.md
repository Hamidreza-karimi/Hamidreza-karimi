#include <iostream>
#include <vector>
#include <cmath>

// Structure to represent a data point
struct DataPoint {
    std::vector<double> features;
    std::string label;
};

// Function to calculate Euclidean distance between two data points
double calculateDistance(const DataPoint& p1, const DataPoint& p2) {
    double distance = 0.0;
    for (size_t i = 0; i < p1.features.size(); ++i) {
        distance += pow(p1.features[i] - p2.features[i], 2);
    }
    return sqrt(distance);
}

// KNN algorithm
std::string knn(const std::vector<DataPoint>& trainingData, const DataPoint& testInstance, int k) {
    std::vector<std::pair<double, std::string>> distances;

    // Calculate distances from test instance to each training instance
    for (const auto& trainingInstance : trainingData) {
        double distance = calculateDistance(trainingInstance, testInstance);
        distances.push_back(std::make_pair(distance, trainingInstance.label));
    }

    // Sort distances in ascending order
    std::sort(distances.begin(), distances.end());

    // Count the labels of the K nearest neighbors
    std::unordered_map<std::string, int> labelCounts;
    for (int i = 0; i < k; ++i) {
        const auto& label = distances[i].second;
        ++labelCounts[label];
    }

    // Find the label with the highest count
    int maxCount = -1;
    std::string predictedLabel;
    for (const auto& count : labelCounts) {
        if (count.second > maxCount) {
            maxCount = count.second;
            predictedLabel = count.first;
        }
    }

    return predictedLabel;
}

int main() {
    // Training data
    std::vector<DataPoint> trainingData = {
        {{1.0, 1.1}, "A"},
        {{1.0, 1.0}, "A"},
        {{0.0, 0.0}, "B"},
        {{0.0, 0.1}, "B"}
    };

    // Test instance
    DataPoint testInstance = {{2.0, 2.0}, ""};

    // Set the value of K
    int k = 3;

    // Apply KNN algorithm
    std::string predictedLabel = knn(trainingData, testInstance, k);

    // Output the predicted label
    std::cout << "Predicted Label: " << predictedLabel << std::endl;

    return 0;
}
