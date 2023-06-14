//This is for informational purposes only
//This is a WIP vectorizer written in C++
//Need Eigen library for sparse matrices

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <Eigen/Sparse>


//Create an unordered hashmap to store our features & the feature column index with num_features
std::unordered_map<std::string, int> feature_indices_;
int num_features_ = 0;

std::vector<std::unordered_map<std::string, std::string>> data0 = {
    {{"color", "orange"}, {"size", "small"}},
    {{"color", "white"}, {"model", "v1.01"}},
    {{"color", "green"}, {"year", "2023"}}
};


void Fit(const std::vector<std::unordered_map<std::string, std::string>>& data) {
    for (const auto& map : data) {

        for (const auto& mapkey : map) {

            std::cout << mapkey.first << " " << mapkey.second << std::endl;

            const std::string& key = mapkey.first;

            auto search = feature_indices_.find(key);

            if (search != feature_indices_.end()) {
                std::cout << "Key " << key << " exists as a feature" << std::endl;
            }
            else {
                std::cout << "Key " << key << " does not exist as a feature. Adding...." << std::endl;

                feature_indices_[key] = num_features_;

                num_features_++;

            };
        };
    };
};

void Transform(const std::vector<std::unordered_map<std::string, std::string>>& data) {
    std::size_t num_samples = data.size();

    std::cout << "num_samples = " << num_samples << std::endl;

    Eigen::SparseMatrix<double> sparseMatrix(num_samples, num_features_);

    std::cout << "Sparse Matrix: " << std::endl;

    std::cout << sparseMatrix << std::endl;


    int index_count = 0;

    for (const auto& hash : data) { 

        std::cout << " Index Count:  " << index_count << std::endl;

        index_count++;

        for (const auto& hashmap : hash) {

            const std::string& first = hashmap.first;

            std::cout << "[" << index_count - 1 << "," << feature_indices_[first] << "]" << std::endl;

            sparseMatrix.insert(index_count - 1, feature_indices_[first]) = 1;

        };

        for (int i = 0; i < sparseMatrix.rows(); ++i) {

            for (int j = 0; j < sparseMatrix.cols(); ++j) {

                std::cout << sparseMatrix.coeff(i, j) << " ";
            };
            std::cout << std::endl;
        };
    };
};

int main() {

    Fit(data0);

    Transform(data0);


    return 0;
}
