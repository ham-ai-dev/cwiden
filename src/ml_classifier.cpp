#include "ml_classifier.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>

// =========================================================================
// Minimal JSON parser (no external dependency)
// =========================================================================
// We only need to parse the specific model.json structure produced by
// train_model.py. This is a purpose-built recursive descent parser.

namespace {

struct JsonValue;

enum class JsonType { Null, Bool, Number, String, Array, Object };

struct JsonValue {
    JsonType type = JsonType::Null;
    double number = 0.0;
    bool boolean = false;
    std::string str;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;

    const JsonValue& operator[](const std::string& key) const {
        for (auto& p : object) {
            if (p.first == key) return p.second;
        }
        static JsonValue null_val;
        return null_val;
    }

    const JsonValue& operator[](int idx) const {
        if (idx >= 0 && idx < static_cast<int>(array.size()))
            return array[idx];
        static JsonValue null_val;
        return null_val;
    }

    int array_size() const { return static_cast<int>(array.size()); }
    int object_size() const { return static_cast<int>(object.size()); }
};

static void skip_ws(const char*& p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
}

static JsonValue parse_value(const char*& p);

static std::string parse_string_raw(const char*& p) {
    if (*p != '"') return "";
    p++; // skip opening quote
    std::string s;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case '"': s += '"'; break;
                case '\\': s += '\\'; break;
                case 'n': s += '\n'; break;
                case 't': s += '\t'; break;
                default: s += *p; break;
            }
        } else {
            s += *p;
        }
        p++;
    }
    if (*p == '"') p++;
    return s;
}

static JsonValue parse_value(const char*& p) {
    skip_ws(p);
    JsonValue v;

    if (*p == '"') {
        v.type = JsonType::String;
        v.str = parse_string_raw(p);
    }
    else if (*p == '{') {
        v.type = JsonType::Object;
        p++; // skip {
        skip_ws(p);
        while (*p && *p != '}') {
            skip_ws(p);
            std::string key = parse_string_raw(p);
            skip_ws(p);
            if (*p == ':') p++;
            JsonValue val = parse_value(p);
            v.object.push_back({key, val});
            skip_ws(p);
            if (*p == ',') p++;
        }
        if (*p == '}') p++;
    }
    else if (*p == '[') {
        v.type = JsonType::Array;
        p++;
        skip_ws(p);
        while (*p && *p != ']') {
            v.array.push_back(parse_value(p));
            skip_ws(p);
            if (*p == ',') p++;
        }
        if (*p == ']') p++;
    }
    else if (*p == 't') { // true
        v.type = JsonType::Bool;
        v.boolean = true;
        p += 4;
    }
    else if (*p == 'f') { // false
        v.type = JsonType::Bool;
        v.boolean = false;
        p += 5;
    }
    else if (*p == 'n') { // null
        v.type = JsonType::Null;
        p += 4;
    }
    else { // number
        v.type = JsonType::Number;
        char* end = nullptr;
        v.number = strtod(p, &end);
        p = end;
    }

    skip_ws(p);
    return v;
}

static JsonValue parse_json(const std::string& text) {
    const char* p = text.c_str();
    return parse_value(p);
}

} // namespace

// =========================================================================
// MLClassifier implementation
// =========================================================================

MLClassifier::MLClassifier() {}
MLClassifier::~MLClassifier() {}

static MLClassifier::Tree parse_tree_node(const JsonValue& node, std::vector<MLClassifier::TreeNode>& nodes) {
    // This is called for the root; we recursively build the flat node array
    (void)node; (void)nodes;
    MLClassifier::Tree t;
    return t;
}

// Recursive tree builder
static int build_tree(const JsonValue& jnode, std::vector<MLClassifier::TreeNode>& nodes) {
    MLClassifier::TreeNode node;

    if (jnode["leaf"].type == JsonType::Number) {
        node.is_leaf = true;
        node.leaf_value = static_cast<float>(jnode["leaf"].number);
        int idx = static_cast<int>(nodes.size());
        nodes.push_back(node);
        return idx;
    }

    node.is_leaf = false;
    node.feature_idx = static_cast<int>(jnode["feature"].number);
    node.threshold = static_cast<float>(jnode["threshold"].number);

    // Reserve our slot
    int my_idx = static_cast<int>(nodes.size());
    nodes.push_back(node); // placeholder

    // Build children
    int left_idx = build_tree(jnode["left"], nodes);
    int right_idx = build_tree(jnode["right"], nodes);

    // Update our node
    nodes[my_idx].left_child = left_idx;
    nodes[my_idx].right_child = right_idx;

    return my_idx;
}

bool MLClassifier::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "MLClassifier: cannot open " << path << std::endl;
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    auto root = parse_json(content);
    if (root.type != JsonType::Object) {
        std::cerr << "MLClassifier: invalid JSON in " << path << std::endl;
        return false;
    }

    // Version check
    int version = static_cast<int>(root["version"].number);
    if (version != 1) {
        std::cerr << "MLClassifier: unsupported model version " << version << std::endl;
        return false;
    }

    learning_rate_ = static_cast<float>(root["learning_rate"].number);
    init_raw_ = static_cast<float>(root["init_raw"].number);

    // Feature names
    feature_names_.clear();
    auto& fnames = root["feature_names"];
    for (int i = 0; i < fnames.array_size(); i++) {
        feature_names_.push_back(fnames[i].str);
    }

    // Feature importances
    importances_.clear();
    auto& fimp = root["feature_importances"];
    for (int i = 0; i < fimp.array_size(); i++) {
        importances_.push_back(static_cast<float>(fimp[i].number));
    }

    // Trees
    trees_.clear();
    auto& jtrees = root["trees"];
    for (int i = 0; i < jtrees.array_size(); i++) {
        Tree tree;
        build_tree(jtrees[i], tree.nodes);
        trees_.push_back(std::move(tree));
    }

    // Stats
    auto& stats = root["stats"];
    n_samples_ = static_cast<int>(stats["n_samples"].number);
    train_accuracy_ = static_cast<float>(stats["train_accuracy"].number);

    loaded_ = true;
    std::cerr << "MLClassifier: loaded " << trees_.size() << " trees from " << path
              << " (trained on " << n_samples_ << " samples, acc=" 
              << static_cast<int>(train_accuracy_ * 100) << "%)" << std::endl;

    return true;
}

float MLClassifier::traverse_tree(const Tree& tree, const std::vector<float>& features) const {
    int idx = 0;
    while (!tree.nodes[idx].is_leaf) {
        auto& n = tree.nodes[idx];
        if (n.feature_idx < static_cast<int>(features.size()) &&
            features[n.feature_idx] <= n.threshold) {
            idx = n.left_child;
        } else {
            idx = n.right_child;
        }
    }
    return tree.nodes[idx].leaf_value;
}

float MLClassifier::predict(const std::vector<float>& features) const {
    if (!loaded_ || trees_.empty()) return 0.5f;

    // GBT prediction: init + sum(learning_rate * tree_i(x))
    float raw = init_raw_;
    for (auto& tree : trees_) {
        raw += learning_rate_ * traverse_tree(tree, features);
    }

    // Sigmoid
    float prob = 1.0f / (1.0f + std::exp(-raw));
    return prob;
}
