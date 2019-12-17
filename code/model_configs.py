MODEL_CONFIGS = {
    "jd":
    {
        "order_feature": 73,
        "POIs_feature": 10,
        "region_feature": 13,

        "G_hidden_features": [50,60],
        "T_hidden_features": {
                                'concate':10,
                                'feature':10,
                                'representation':5,
                             },
        "S_hidden_features": [10,5],

        "G_output_features": 73,
        "S_output_features": 5,
        "T_output_features": 5,

        "support_feature": 73,

        "sample_L": 10,
    },

    "7fresh":
    {
        "order_feature": 73,
        "POIs_feature": 10,
        "region_feature": 13,

        "G_hidden_features": [50,60],
        "T_hidden_features": {
                                'concate':10,
                                'feature':10,
                                'representation':5,
                             },
        "S_hidden_features": [10,5],

        "G_output_features": 73,
        "S_output_features": 5,
        "T_output_features": 5,

        "support_feature": 73,

        "sample_L": 10,
    },
}