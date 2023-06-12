#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>


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
                std::cout << "Key " << key << " does not exist as a feature" << std::endl;

                feature_indices_[key] = num_features_;

                num_features_++;

            };
        };
    };
};

int main() {

    Fit(data0);

    for (const auto& dict : feature_indices_) {

        std::cout << "Key  " << dict.first << " Value  " << dict.second << std::endl;
    }

    return 0;
}
