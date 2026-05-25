#pragma once
/**
 * ml_classifier.hpp — Lightweight ML model inference for CW classification.
 *
 * Loads a gradient-boosted tree ensemble from model.json (produced by
 * scripts/train_model.py) and runs inference in pure C++.
 * No Python/sklearn dependency at runtime.
 */

#include <string>
#include <vector>

class MLClassifier {
public:
    MLClassifier();
    ~MLClassifier();

    /** Load model from JSON file. Returns true on success. */
    bool load(const std::string& path);

    /** Is a model loaded and ready? */
    bool is_loaded() const { return loaded_; }

    /** Number of training samples used. */
    int training_samples() const { return n_samples_; }
    float training_accuracy() const { return train_accuracy_; }

    /**
     * Predict CW probability from feature vector.
     *
     * @param features  Vector of features in standard order:
     *   [snr_db, effective_bw, shape_factor, headroom_db,
     *    bimodality_coeff, on_off_ratio, rhythm_score, wpm_estimate,
     *    spectral_entropy, peak_stability]
     * @return Probability of CW (0.0 to 1.0)
     */
    float predict(const std::vector<float>& features) const;

    /** Feature names expected by this model. */
    const std::vector<std::string>& feature_names() const { return feature_names_; }

    /** Feature importance scores (same order as feature_names). */
    const std::vector<float>& feature_importances() const { return importances_; }

    struct TreeNode {
        bool is_leaf = false;
        float leaf_value = 0.0f;
        int feature_idx = -1;
        float threshold = 0.0f;
        int left_child = -1;
        int right_child = -1;
    };

    struct Tree {
        std::vector<TreeNode> nodes;
    };

private:

    float traverse_tree(const Tree& tree, const std::vector<float>& features) const;

    bool loaded_ = false;
    float learning_rate_ = 0.1f;
    float init_raw_ = 0.0f;  // initial log-odds prediction
    std::vector<Tree> trees_;
    std::vector<std::string> feature_names_;
    std::vector<float> importances_;
    int n_samples_ = 0;
    float train_accuracy_ = 0.0f;
};
