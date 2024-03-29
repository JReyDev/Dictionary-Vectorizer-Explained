//This is for informational purposes only
//This is a WIP vectorizer written in C++
//Need Eigen library for sparse matrices

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <Eigen/Sparse>


//Sample Data
std::vector<std::unordered_map<std::string, std::string>> data0 = {
    {{"color", "orange"}, {"size", "small"}},
    {{"color", "white"}, {"model", "v1.01"}},
    {{"color", "green"}, {"year", "2023"}}
};

class SampleHashVectorizer {

public:

    // Create an unordered hashmap to store our features & the feature column index with num_features

    std::unordered_map<std::string, int> feature_indices_;

    int num_features_ = 0;

    void Fit(const std::vector<std::unordered_map<std::string, std::string>>& data) {

        for (const auto& map : data) {

            for (const auto& mapkey : map) {

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

        Eigen::SparseMatrix<double> sparseMatrix(num_samples, num_features_);

        int index_count = 0;

        for (const auto& hash : data) { 

            index_count++;

            for (const auto& hashmap : hash) {

                const std::string& first = hashmap.first;

                std::cout << "[" << index_count - 1 << "," << feature_indices_[first] << "]" << std::endl;

                sparseMatrix.coeffRef(index_count - 1, feature_indices_[first]) = 1;

            };

            for (int i = 0; i < sparseMatrix.rows(); ++i) {

                for (int j = 0; j < sparseMatrix.cols(); ++j) {

                    std::cout << sparseMatrix.coeff(i, j) << " ";
                };
                std::cout << std::endl;
                };
            };
        };
    };

int main() {

    SampleHashVectorizer HashVec;

    HashVec.Fit(data0);

    HashVec.Transform(data0);


    return 0;
}
